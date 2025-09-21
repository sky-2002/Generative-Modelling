from dataclasses import dataclass
from torch import nn
import torch
from typing import Optional


@dataclass
class DeepSeekModelConfig:
    num_attention_heads: int = 8
    input_dim: int = 512
    embed_dim: int = 512
    bias: bool = False
    use_mla: bool = False

    kv_heads: int = (
        4  # number of groups of attention heads that share the same K and V matrices
    )

    kv_latent_dim: int = 4
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


if __name__ == "__main__":
    x = torch.rand(1, 2, 3)
    config = DeepSeekModelConfig()
    mha = MultiHeadAttention(config)
    mqa = MultiQueryAttention(config)
    gqa = GroupedQueryAttention(config)

    print(sum(p.numel() for p in mha.parameters()))
    print(sum(p.numel() for p in mqa.parameters()))
    print(sum(p.numel() for p in gqa.parameters()))
