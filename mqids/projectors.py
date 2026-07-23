"""Continuous-token projectors from MOIRAI space into Qwen embedding space."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source_dtype = value.dtype
        value_float = value.float()
        value_float = value_float * torch.rsqrt(value_float.square().mean(-1, keepdim=True) + self.eps)
        return (value_float.to(source_dtype) * self.weight.to(source_dtype))


class LinearProjector(nn.Module):
    def __init__(self, d_moirai: int, d_llm: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_moirai)
        self.projection = nn.Linear(d_moirai, d_llm)
        self.output_norm = RMSNorm(d_llm)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.projection(self.input_norm(tokens)))


class DirectProjector(nn.Module):
    """Competitive nonlinear baseline: expand, activate, then map to Qwen."""

    def __init__(
        self,
        d_moirai: int,
        d_llm: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_moirai)
        self.net = nn.Sequential(
            nn.Linear(d_moirai, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_llm),
        )
        self.output_norm = RMSNorm(d_llm)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.net(self.input_norm(tokens)))


class ReprogrammingAttention(nn.Module):
    """Cross-attention adapted from Time-LLM's ReprogrammingLayer.

    Time-LLM copyright (c) 2024; distributed under Apache-2.0. This version
    exposes attention weights and accepts a compact, fixed prototype bank rather
    than learning a dense projection over the full LLM vocabulary.
    """

    def __init__(
        self,
        d_moirai: int,
        d_llm: int,
        attention_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim % n_heads:
            raise ValueError("attention_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = attention_dim // n_heads
        self.query_projection = nn.Linear(d_moirai, attention_dim)
        self.key_projection = nn.Linear(d_llm, attention_dim)
        self.value_projection = nn.Linear(d_llm, attention_dim)
        self.out_projection = nn.Linear(attention_dim, d_llm)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_tokens: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = target_tokens.shape
        prototype_count = prototypes.shape[0]
        query = self.query_projection(target_tokens).view(batch, length, self.n_heads, self.head_dim)
        key = self.key_projection(prototypes).view(prototype_count, self.n_heads, self.head_dim)
        value = self.value_projection(prototypes).view(prototype_count, self.n_heads, self.head_dim)
        scores = torch.einsum("blhe,phe->bhlp", query, key) / sqrt(self.head_dim)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.einsum("bhlp,phe->blhe", attention, value).reshape(batch, length, -1)
        return self.out_projection(output), attention


class ReprogrammingProjector(nn.Module):
    """Direct information path plus gated semantic prototype reprogramming."""

    def __init__(
        self,
        d_moirai: int,
        d_llm: int,
        hidden_dim: int,
        attention_dim: int,
        n_heads: int,
        prototypes: torch.Tensor,
        dropout: float = 0.1,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if prototypes.ndim != 2 or prototypes.shape[1] != d_llm:
            raise ValueError(f"Expected prototypes [count, {d_llm}], got {tuple(prototypes.shape)}")
        self.register_buffer("prototypes", prototypes.detach().clone(), persistent=True)
        self.input_norm = nn.LayerNorm(d_moirai)
        self.direct = DirectProjector(d_moirai, d_llm, hidden_dim, dropout)
        self.reprogramming = ReprogrammingAttention(
            d_moirai=d_moirai,
            d_llm=d_llm,
            attention_dim=attention_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate)).logit())
        self.output_norm = RMSNorm(d_llm)
        self.last_attention: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(tokens)
        direct = self.direct(tokens)
        semantic, attention = self.reprogramming(normalized, self.prototypes.to(tokens.dtype))
        self.last_attention = attention.detach()
        gate = torch.sigmoid(self.gate_logit)
        return self.output_norm(direct + gate * semantic)
