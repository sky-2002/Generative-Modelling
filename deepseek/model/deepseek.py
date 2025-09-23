from dataclasses import dataclass
from torch import nn
import torch
from typing import Optional


@dataclass
class DeepSeekModelConfig:
    num_attention_heads: int = 8
    input_dim: int = 4
    embed_dim: int = 32
    bias: bool = False

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
    pass


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
            Qr = self.rope.apply_rope(Q)
        else:
            Qr = self.rope.apply_rope(
                self.Wqr(query_latent)
                .view(batch_size, num_tokens, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        # ---------------------

        # ----- KV latent -----
        kv_latent = self.Wdkv(x)  # [B, T, kv_latent_dim]
        # update cache
        self.kv_latent_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv_latent)

        kv_latent_all = self.kv_latent_cache[
            :batch_size, :end_pos
        ]  # [B, T, kv_latent_dim]

        # [B, num_heads, T, head_dim] x [head_dim, kv_latent_dim]
        Q_absorbed = Q @ self.Wuk.weight.T  # B, num_heads, T, kv_latent_dim

        V = self.Wuv(kv_latent_all).view(
            batch_size, S, self.mla_kv_heads, self.head_dim
        )  # [B, S, mla_kv_heads, head_dim]
        # expand V to match n_heads
        V = V.repeat_interleave(
            self.num_heads // self.mla_kv_heads, dim=2
        )  # [B, T, num_heads, head_dim]

        V = V.transpose(1, 2)  # [B, H, S, D]

        # ----- RoPE path -----
        K_pos_encoding = self.rope.apply_rope(self.Wkr(x)).view(
            batch_size, num_tokens, self.mla_kv_heads, self.head_dim
        )  # B, T, mla_kv_heads, head_dim
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

        # ----- Context -----
        context = attention_weights @ V  # [B, H, T, D]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.embed_dim)
        )
        out = self.out_proj(context)
        return out, self.kv_latent_cache


if __name__ == "__main__":
    config = DeepSeekModelConfig()
    x = torch.rand(1, 2, config.input_dim)
    mha = MultiHeadAttention(config)
    mqa = MultiQueryAttention(config)
    gqa = GroupedQueryAttention(config)
    mhla = MultiHeadLatentAttention(config)

    print(sum(p.numel() for p in mha.parameters()))
    print(sum(p.numel() for p in mqa.parameters()))
    print(sum(p.numel() for p in gqa.parameters()))
    print(sum(p.numel() for p in mhla.parameters()))
