from dataclasses import dataclass
from torch import nn
import torch


@dataclass
class DeepSeekModelConfig:
    num_attention_heads: int = 8
    input_dim: int = 512
    embed_dim: int = 512
    bias: bool = False
    use_mla: bool = False
    pass


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


if __name__ == "__main__":
    x = torch.rand(1, 2, 3)
    config = DeepSeekModelConfig()
    mha = MultiHeadAttention(config)
    mqa = MultiQueryAttention(config)

    print(sum(p.numel() for p in mha.parameters()))
    print(sum(p.numel() for p in mqa.parameters()))
