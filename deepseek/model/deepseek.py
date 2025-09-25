from dataclasses import dataclass
from torch import nn
import torch
from typing import Optional
import torch.nn.functional as F


@dataclass
class DeepSeekModelConfig:
    num_attention_heads: int = 8
    input_dim: int = 1024
    embed_dim: int = 1024
    bias: bool = False
    dropout: float = 0.1

    kv_heads: int = 4  # number of key-value heads for grouped query attention

    # configs needed for MLA
    mla_kv_heads: int = (
        4  # number of groups of attention heads that share the same K and V matrices
    )
    use_mla: bool = False
    num_gpus: int = 1  # number of gpus
    # n_local_heads
    # this is maybe for cases where computation is distributed across gpus, will have to read more

    q_latent_dim: int = 4  # dimension of latent used to build queries
    kv_latent_dim: int = 4  # dimension of latent used to build keys and values

    # in official implementation, there are configs for
    # rope and no-rope attention head dimensions, I am keeping it same as head dim
    # since we concatenate the no-rope and rope queries and keys, they add these dimnensions
    # to be later used to scaling attention scores

    max_batch_size: int = 8
    max_token_len: int = 1024

    num_shared_experts: int = 8
    num_routed_experts: int = 16
    moe_top_k: int = 2
    expert_intermediate_dim: int = 8192

    num_dense_ffn: int = 2
    num_moe_ffn: int = 4

    mtp_depth: int = 3
    vocab_size: int = 50257


class Expert(nn.Module):

    def __init__(self, input_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, intermediate_dim)
        self.w11 = nn.Linear(input_dim, intermediate_dim)
        self.w2 = nn.Linear(intermediate_dim, input_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w11(x)))


class MoE(nn.Module):
    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.num_local_experts = config.num_routed_experts // config.num_gpus
        self.top_k = config.moe_top_k

        self.expert_selector = nn.Linear(
            config.input_dim, self.num_routed_experts, bias=False
        )
        self.routed_experts = nn.ModuleList(
            [
                Expert(config.input_dim, config.expert_intermediate_dim)
                for _ in range(self.num_routed_experts)
            ]
        )
        self.shared_experts = Expert(
            config.input_dim, config.expert_intermediate_dim * self.num_shared_experts
        )

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        gate_output, topk_indices = self.topk_routing(x)
        x = x.view(
            batch_size * num_tokens, input_dim
        )  # so now it is like a list of tokens
        gate_output = gate_output.view(batch_size * num_tokens, -1)

        topk_indices = topk_indices.view(batch_size * num_tokens, -1)

        y = torch.zeros_like(x)
        counts = torch.bincount(
            topk_indices.flatten(), minlength=self.num_routed_experts
        ).tolist()
        for i in range(self.num_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.routed_experts[i]

            idx, expert_rank = torch.where(topk_indices == i)
            y[idx] += expert(x[idx]) * gate_output[idx, expert_rank, None]

        z = self.shared_experts(x)
        return (y + z).view(batch_size, num_tokens, input_dim)

    def topk_routing(self, x, bias=None):
        batch_size, num_tokens, input_dim = x.shape

        expert_logits = self.expert_selector(x)  # B, T, num_experts
        if bias:
            expert_logits = expert_logits + bias
        topk_logits, topk_indices = torch.topk(expert_logits, k=self.top_k, dim=-1)
        zeros = torch.full_like(expert_logits, float("-inf"))
        sparse_logits = zeros.scatter(dim=-1, index=topk_indices, src=topk_logits)
        gate_output = sparse_logits.softmax(dim=-1)
        return gate_output, topk_indices


class RoPE(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0

    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        if seq_len > self._cached_seq_len or self._cached_cos is None:

            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

            freqs = torch.outer(t, self.inv_freq)

            cos_vals = torch.cos(freqs)
            sin_vals = torch.sin(freqs)

            self._cached_cos = cos_vals
            self._cached_sin = sin_vals
            self._cached_seq_len = seq_len

        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]

    def apply_rope(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """Apply RoPE to input tensor"""
        batch_size, num_tokens, n_heads, head_dim = x.shape

        cos, sin = self._compute_cos_sin(num_tokens, x.device)

        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]

        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(2)

        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

        return rotated_x


class MultiHeadAttention(nn.Module):
    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        self.Wq = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.Wv = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.input_dim, bias=config.bias)

    def forward(self, x):
        # x is B, T, input_dim
        batch_size, num_tokens, input_dim = x.shape
        Q = (
            self.Wq(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # becomes B, num_heads, T, head_dim
        K = (
            self.Wk(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # becomes B, num_heads, T, head_dim
        V = (
            self.Wv(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # becomes B, num_heads, T, head_dim

        attention_scores = Q @ K.transpose(2, 3)
        attention_scores = attention_scores / (self.head_dim**0.5)

        causal_mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)

        attention_scores = attention_scores.masked_fill(
            causal_mask.bool(), float("-inf")
        )
        attention_weights = torch.softmax(
            attention_scores, dim=-1
        )  # B, num_heads, T, T

        context = attention_weights @ V  # B, num_heads, T, head_dim
        context = attention_weights.transpose(1, 2)  # B, T, num_heads, head_dim
        context = attention_weights.view(batch_size, num_tokens, self.embed_dim)
        out = self.out_proj(context)  # B, T, input_dim
        return out


class MultiQueryAttention(nn.Module):
    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        self.Wq = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.input_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(self.input_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.input_dim, bias=config.bias)

    def forward(self, x):
        # x is B, T, input_dim
        batch_size, num_tokens, input_dim = x.shape
        Q = (
            self.Wq(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # becomes B, num_heads, T, head_dim
        K = self.Wk(x)  # B, T, head_dim
        V = self.Wv(x)  # B, T, head_dim

        # create copies for all heads
        K = K.expand(-1, self.num_heads, -1, -1)
        V = V.expand(-1, self.num_heads, -1, -1)

        attention_scores = Q @ K.transpose(2, 3)
        attention_scores = attention_scores / (self.head_dim**0.5)

        causal_mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)

        attention_scores = attention_scores.masked_fill(
            causal_mask.bool(), float("-inf")
        )
        attention_weights = torch.softmax(
            attention_scores, dim=-1
        )  # B, num_heads, T, T

        context = attention_weights @ V  # B, num_heads, T, head_dim
        context = attention_weights.transpose(1, 2)  # B, T, num_heads, head_dim
        context = attention_weights.view(batch_size, num_tokens, self.embed_dim)
        out = self.out_proj(context)  # B, T, input_dim
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = config.kv_heads

        self.Wq = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.input_dim, self.head_dim * config.kv_heads, bias=False)
        self.Wv = nn.Linear(self.input_dim, self.head_dim * config.kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.input_dim, bias=config.bias)

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        Q = (
            self.Wq(x)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # becomes B, num_heads, T, head_dim

        K = self.Wk(x)  # B, T, head_dim*kv_heads
        V = self.Wv(x)  # B, T, head_dim*kv_heads

        K = K.view(batch_size, num_tokens, self.kv_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.kv_heads, self.head_dim)

        # now i need this
        # if kv_heads is 3 and num_heads is 6
        # I want k = [k1, k1, k2, k2, k3, k3] and same for v
        K = K.repeat_interleave(
            self.num_heads // self.kv_heads, dim=2
        )  # B, T, num_heads, head_dim
        V = V.repeat_interleave(
            self.num_heads // self.kv_heads, dim=2
        )  # B, T, num_heads, head_dim

        attention_scores = Q @ K.transpose(2, 3)
        attention_scores = attention_scores / (self.head_dim**0.5)

        causal_mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)

        attention_scores = attention_scores.masked_fill(
            causal_mask.bool(), float("-inf")
        )
        attention_weights = torch.softmax(
            attention_scores, dim=-1
        )  # B, num_heads, T, T

        context = attention_weights @ V  # B, num_heads, T, head_dim
        context = attention_weights.transpose(1, 2)  # B, T, num_heads, head_dim
        context = attention_weights.view(batch_size, num_tokens, self.embed_dim)
        out = self.out_proj(context)  # B, T, input_dim
        return out


# I have copied RMSNorm directly from Deepseek-V3 repo
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


# TODO:
# 1. Try out grouped query attention styled MLA, where each kv head has its own latent cache
# 2.Try out sliding window attention, I read about this in gemma paper
class MultiHeadLatentAttention(nn.Module):

    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.n_local_heads = config.num_attention_heads // config.num_gpus
        self.head_dim = self.embed_dim // self.num_heads
        self.mla_kv_heads = config.mla_kv_heads
        self.kv_latent_dim = config.kv_latent_dim
        self.q_latent_dim = config.q_latent_dim
        self.dropout = nn.Dropout(config.dropout)

        self.rope = RoPE(dim=self.head_dim)
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.input_dim, bias=False
        )

        if self.q_latent_dim == 0:
            self.Wq = nn.Linear(
                self.input_dim, self.num_heads * self.head_dim, bias=False
            )
        else:
            # -------------------(decoupled from RoPE)-----------------------------
            # Query path - This feels to me like LoRa on Q
            # because instead of Wq (input_dim, input_dim) we now have
            # Wdq(input_dim, q_latent_dim) and Wuq(q_latent_dim, input_dim)
            self.Wdq = nn.Linear(self.input_dim, self.q_latent_dim, bias=False)
            self.q_norm = RMSNorm(self.q_latent_dim)
            self.Wuq = nn.Linear(
                self.q_latent_dim, self.num_heads * self.head_dim, bias=False
            )

        # this will build KV latent and also construct K and V from it
        self.Wdkv = nn.Linear(self.input_dim, self.kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_latent_dim)
        self.Wuk = nn.Linear(
            self.kv_latent_dim, self.head_dim, bias=False
        )  # here I am not using num_heads because we will use kv heads (grouped query attention)
        self.Wuv = nn.Linear(
            self.kv_latent_dim, self.mla_kv_heads * self.head_dim, bias=False
        )

        # cache the kv latent and the roped keys
        self.register_buffer(
            "kv_latent_cache",
            torch.zeros(
                config.max_batch_size, config.max_token_len, self.kv_latent_dim
            ),
            persistent=False,  # I won't store on disk
        )
        self.register_buffer(
            "keys_roped",
            torch.zeros(
                config.max_batch_size,
                config.max_token_len,
                self.mla_kv_heads,
                # I could have not used these heads, then we have same keys for each head,4
                # here it is same for a group of attention heads which come under one kv head
                self.head_dim,
            ),
            persistent=False,
        )
        # --------------------------------------------------------------------

        # -------------RoPE path----------------------------------------------
        self.Wkr = nn.Linear(
            self.input_dim, self.mla_kv_heads * self.head_dim, bias=False
        )
        self.Wqr = nn.Linear(self.q_latent_dim, self.embed_dim, bias=False)

    def forward(self, x, start_pos=0):
        batch_size, num_tokens, input_dim = x.shape
        end_pos = start_pos + num_tokens
        S = end_pos  # total cached sequence length

        # ----- Queries -----
        if self.q_latent_dim == 0:
            Q = (
                self.Wq(x)
                .view(batch_size, num_tokens, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )  # [B, num_heads, T, head_dim]
        else:
            query_latent = self.Wdq(x)
            query_latent = self.q_norm(query_latent)
            Q = (
                self.Wuq(query_latent)
                .view(batch_size, num_tokens, self.num_heads, self.head_dim)
                .transpose(1, 2)  # [B, num_heads, T, head_dim]
            )
        # ----- RoPE path -----
        if self.q_latent_dim == 0:
            Qr = self.rope.apply_rope(
                Q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
            ).transpose(1, 2)
        else:
            Qr = self.rope.apply_rope(
                self.Wqr(query_latent).view(
                    batch_size, num_tokens, self.num_heads, self.head_dim
                )
            ).transpose(1, 2)
        # ---------------------

        # ----- KV latent -----
        kv_latent = self.Wdkv(x)  # [B, T, kv_latent_dim]
        # update cache
        self.kv_latent_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv_latent)

        kv_latent_all = self.kv_latent_cache[
            :batch_size, :end_pos
        ]  # [B, T, kv_latent_dim]

        # [B, num_heads, T, head_dim] x [head_dim, kv_latent_dim]
        Q_absorbed = Q @ self.Wuk.weight  # B, num_heads, T, kv_latent_dim

        V = self.Wuv(kv_latent_all).view(
            batch_size, S, self.mla_kv_heads, self.head_dim
        )  # [B, S, mla_kv_heads, head_dim]
        # expand V to match n_heads
        V = V.repeat_interleave(
            self.num_heads // self.mla_kv_heads, dim=2
        )  # [B, T, num_heads, head_dim]

        V = V.transpose(1, 2)  # [B, H, S, D]

        # ----- RoPE path -----
        K_pos_encoding = self.rope.apply_rope(
            self.Wkr(x)
            .view(batch_size, num_tokens, self.mla_kv_heads, self.head_dim)
            .transpose(1, 2)
        ).transpose(
            1, 2
        )  # B, T, mla_kv_heads head_dim
        self.keys_roped[:batch_size, start_pos:end_pos] = K_pos_encoding
        keys_roped_all = self.keys_roped[:batch_size, :end_pos]
        Kr = (
            keys_roped_all.repeat_interleave(self.num_heads // self.mla_kv_heads, dim=2)
            .view(batch_size, S, self.num_heads, self.head_dim)
            .transpose(1, 2)  # [B, S, T, head_dim]
        )

        # ----- Attention scores -----
        # doing unsqueeze to account for heads, since kv cache is only one, not per head
        attention_scores_1 = Q_absorbed @ kv_latent_all.unsqueeze(1).transpose(2, 3)

        attention_scores_2 = Qr @ Kr.transpose(-2, -1)  # [B, num_heads, T, T]
        attention_scores = (attention_scores_1 + attention_scores_2) / (
            2 * self.head_dim
        ) ** 0.5

        # causal mask
        causal_mask = torch.triu(
            torch.ones(end_pos, end_pos, device=x.device), diagonal=1
        )
        attention_scores = attention_scores.masked_fill(
            causal_mask.bool()[:, -num_tokens:], float("-inf")
        )

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # ----- Context -----
        context = attention_weights @ V  # [B, H, T, D]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.embed_dim)
        )
        out = self.out_proj(context)
        return out


# Note: I might not use this in training, will do normal single token prediction only
class BasicMultiTokenPrediction(nn.Module):

    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()

        # If k is mtp_depth, and current token position is i
        # this module predicts next k tokens, so from
        # (i+1) to (i+k)
        self.k = config.mtp_depth
        self.vocab_size = config.vocab_size
        self.rms_norm = RMSNorm(config.input_dim)
        self.embed = nn.Embedding(self.vocab_size, config.input_dim)
        self.unembed = nn.Linear(config.input_dim, self.vocab_size, bias=False)
        self.unembed.weight = self.embed.weight

        self.projections = nn.ModuleList(
            [nn.Linear(2 * config.input_dim, config.input_dim) for _ in range(self.k)]
        )

        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(config.input_dim, config.num_attention_heads)
                for _ in range(self.k)
            ]
        )

    def forward(self, x):
        # x is the final hidden states for all tokens that we get after all transformer blocks,
        # so it is just before the final un-ebedding layer
        batch_size, num_tokens, input_size = x.shape
        # if num_tokens is 6
        # i = 0, 1, 2, 3, 4, 5
        # k=3
        # i can predict till 2+3 = 5
        # so i have to iterate i from 0 to 2 only
        # 2 = 6(num_tokens)-3(k)-1
        # so I have to go till x[:,num_tokens-k, :]

        logits = []

        for ith_token_pos in range(0, num_tokens - self.k):
            hidden_state_ith_token = x[:, ith_token_pos, :]

            logits_k = []
            for k in range(self.k):

                future_position = ith_token_pos + k + 1
                token_embedding = x[
                    :, future_position, :
                ]  # considering x as the final hidden state after all blocks

                _h = self.rms_norm(hidden_state_ith_token)
                _e = self.rms_norm(token_embedding)
                merged = torch.cat([_h, _e], dim=1)

                proj = self.projections[k](merged).unsqueeze(0)
                out = self.transformers[k](proj)
                hidden_state_current = out.squeeze(0)
                _logits = self.unembed(hidden_state_current)
                logits_k.append(_logits)

                hidden_state_ith_token = hidden_state_current

            logits_k = torch.stack(logits_k, dim=1)
            logits.append(logits_k)

        logits = torch.stack(logits, dim=0)
        logits = logits.permute(1, 0, 2, 3).contiguous()
        return logits


class TransformerBlock(nn.Module):

    def __init__(self, config: DeepSeekModelConfig, moe: bool = True):
        super().__init__()
        self.rms_norm_1 = RMSNorm(config.input_dim)
        self.mhla = MultiHeadLatentAttention(config)
        self.rms_norm_2 = RMSNorm(config.input_dim)

        if moe:
            self.ffn = MoE(config)
        else:
            self.ffn = Expert(config.input_dim, config.expert_intermediate_dim)

    def forward(self, x):
        x = x + self.mhla(self.rms_norm_1(x))
        x = x + self.ffn(self.rms_norm_2(x))
        return x


class DeepseekInspiredModel(nn.Module):
    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.input_dim)
        self.position_embedding = nn.Embedding(config.max_token_len, config.input_dim)

        _blocks = [
            TransformerBlock(config, moe=False) for _ in range(config.num_dense_ffn)
        ]
        _blocks.extend(
            [TransformerBlock(config, moe=True) for _ in range(config.num_moe_ffn)]
        )
        self.transformer_blocks = nn.ModuleList(_blocks)

        self.ln_f = RMSNorm(config.input_dim)
        self.head = nn.Linear(config.input_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, x):
        batch_size, num_tokens = x.shape

        token_embeddings = self.token_embedding(x)
        position_ids = torch.arange(0, num_tokens, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        h = token_embeddings + position_embeddings

        for block in self.transformer_blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


if __name__ == "__main__":
    config = DeepSeekModelConfig()
    x = torch.rand(1, 10)

    dim = DeepseekInspiredModel(config)

    print(
        f"Number of parameters (in millions): {sum(p.numel() for p in dim.parameters()) / 1_000_000}"
    )
    print(
        f"Number of parameters (in GB): {sum(p.numel() for p in dim.parameters())*4/1024**3:.2f} GB"
    )
