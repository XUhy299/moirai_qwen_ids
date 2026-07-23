"""Frozen-MOIRAI/frozen-Qwen classifier with head and verbalizer outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import DiscreteStateVocabulary
from .prompting import PromptAssembler, single_token_label_ids
from .projectors import RMSNorm
from .switch_templates import build_discrete_text


@dataclass
class ClassifierOutput:
    classifier_logits: torch.Tensor
    verbalizer_logits: torch.Tensor | None
    moirai_tokens: torch.Tensor
    soft_tokens: torch.Tensor
    query_hidden: torch.Tensor


class MoiraiQwenClassifier(nn.Module):
    def __init__(
        self,
        *,
        moirai_tokenizer: nn.Module,
        qwen_causal_lm: nn.Module,
        tokenizer,
        projector: nn.Module,
        variable_type_ids: torch.Tensor,
        labels: tuple[str, str] = ("正常", "异常"),
        prompt_assembler: PromptAssembler | None = None,
        discrete_to_text: bool = False,
        active_names: tuple[str, ...] = (),
        active_descriptions: tuple[str, ...] = (),
        semantic_style: str = "compact",
        discrete_vocabulary: DiscreteStateVocabulary | None = None,
        continuous_indices: tuple[int, ...] = (),
        discrete_indices: tuple[int, ...] = (),
        window_length: int = 0,
    ) -> None:
        super().__init__()
        self.moirai_tokenizer = moirai_tokenizer
        self.qwen = qwen_causal_lm
        self.projector = projector
        self.prompt = prompt_assembler or PromptAssembler(tokenizer)
        self.discrete_to_text = discrete_to_text
        self.active_names = active_names
        self.active_descriptions = active_descriptions
        self.semantic_style = semantic_style
        self.discrete_vocabulary = discrete_vocabulary
        self.window_length = int(window_length)
        self.continuous_indices = tuple(int(index) for index in continuous_indices)
        self.discrete_indices = tuple(int(index) for index in discrete_indices)

        if discrete_to_text:
            if discrete_vocabulary is None:
                raise ValueError("DTT requires a train-derived discrete state vocabulary")
            if len(active_names) != int(variable_type_ids.numel()):
                raise ValueError("DTT active names must match variable metadata")
            if len(active_descriptions) != len(active_names):
                raise ValueError("DTT descriptions must match active names")
            if self.window_length <= 0:
                raise ValueError("DTT requires a positive window length")

        self.register_buffer(
            "_continuous_indices",
            torch.tensor(continuous_indices, dtype=torch.long) if continuous_indices else torch.empty(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_discrete_indices",
            torch.tensor(discrete_indices, dtype=torch.long) if discrete_indices else torch.empty(0, dtype=torch.long),
            persistent=True,
        )

        if discrete_to_text:
            self.num_variables = len(continuous_indices)
        else:
            self.num_variables = int(variable_type_ids.numel())

        self.d_llm = int(self.qwen.config.hidden_size)
        self.variable_embedding = nn.Embedding(self.num_variables, self.d_llm)
        type_count = max(int(variable_type_ids.max().item()) + 1, 3)
        self.type_embedding = nn.Embedding(type_count, self.d_llm)
        self.register_buffer("variable_type_ids", variable_type_ids.long(), persistent=True)
        self.register_buffer(
            "label_token_ids",
            torch.tensor(single_token_label_ids(tokenizer, labels), dtype=torch.long),
            persistent=True,
        )
        self.fusion_norm = RMSNorm(self.d_llm)
        self.classifier_norm = RMSNorm(self.d_llm)
        self.classifier = nn.Linear(self.d_llm, 2)
        nn.init.normal_(self.variable_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)
        self.qwen.eval()

    def train(self, mode: bool = True) -> "MoiraiQwenClassifier":
        super().train(mode)
        self.qwen.eval()
        self.moirai_tokenizer.eval()
        return self

    def forward(self, windows: torch.Tensor) -> ClassifierOutput:
        discrete_states: list[tuple[str, ...]] | None = None

        if self.discrete_to_text:
            windows_cont = windows[..., self._continuous_indices]
            windows_disc = windows[..., self._discrete_indices]
            moirai_tokens = self.moirai_tokenizer(windows_cont)
            if moirai_tokens.shape[1] != self.num_variables:
                raise ValueError("MOIRAI token count does not match continuous variable count")
            soft_tokens = self.projector(moirai_tokens)
            variable_ids = torch.arange(self.num_variables, device=soft_tokens.device)
            var_types = self.variable_type_ids[self._continuous_indices]
            metadata = self.variable_embedding(variable_ids) + self.type_embedding(var_types)
            soft_tokens = self.fusion_norm(soft_tokens + metadata.unsqueeze(0))

            if self._discrete_indices.numel() > 0:
                batch_size = windows_disc.shape[0]
                discrete_states = [
                    build_discrete_text(
                        windows_disc[i].T,
                        self.discrete_vocabulary,
                    )
                    for i in range(batch_size)
                ]
        else:
            moirai_tokens = self.moirai_tokenizer(windows)
            if moirai_tokens.shape[1] != self.num_variables:
                raise ValueError("MOIRAI token count does not match variable metadata")
            soft_tokens = self.projector(moirai_tokens)
            variable_ids = torch.arange(self.num_variables, device=soft_tokens.device)
            metadata = self.variable_embedding(variable_ids) + self.type_embedding(self.variable_type_ids)
            soft_tokens = self.fusion_norm(soft_tokens + metadata.unsqueeze(0))

        embedding_layer = self.qwen.get_input_embeddings()
        qwen_dtype = embedding_layer.weight.dtype
        if self.discrete_to_text:
            if discrete_states is None:
                raise ValueError("DTT did not produce per-sample endpoint states")
            inputs_embeds, attention_mask = self.prompt.forward_variable_aligned(
                soft_tokens.to(qwen_dtype),
                embedding_layer,
                active_names=self.active_names,
                active_descriptions=self.active_descriptions,
                semantic_style=self.semantic_style,
                continuous_indices=self.continuous_indices,
                discrete_indices=self.discrete_indices,
                discrete_states=discrete_states,
                window_length=self.window_length,
            )
        else:
            inputs_embeds, attention_mask = self.prompt(
                soft_tokens.to(qwen_dtype),
                embedding_layer,
            )
        backbone = getattr(self.qwen, "model", None)
        if backbone is None:
            raise TypeError("Expected a causal-LM wrapper exposing its decoder as `.model`")
        result = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        query_hidden = result.last_hidden_state[:, -1]
        classifier_logits = self.classifier(self.classifier_norm(query_hidden).float())
        output_embedding = self.qwen.get_output_embeddings().weight
        label_weights = output_embedding.index_select(0, self.label_token_ids)
        verbalizer_logits = F.linear(query_hidden, label_weights)
        return ClassifierOutput(
            classifier_logits=classifier_logits,
            verbalizer_logits=verbalizer_logits,
            moirai_tokens=moirai_tokens,
            soft_tokens=soft_tokens,
            query_hidden=query_hidden,
        )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class MoiraiEncoderClassifier(nn.Module):
    """No-LLM baseline with a small cross-variable Transformer encoder."""

    def __init__(
        self,
        *,
        moirai_tokenizer: nn.Module,
        d_moirai: int,
        variable_type_ids: torch.Tensor,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("Baseline hidden_dim must be divisible by heads")
        self.moirai_tokenizer = moirai_tokenizer
        self.num_variables = int(variable_type_ids.numel())
        self.input_projection = nn.Sequential(
            nn.LayerNorm(d_moirai),
            nn.Linear(d_moirai, hidden_dim),
            nn.GELU(),
        )
        self.variable_embedding = nn.Embedding(self.num_variables, hidden_dim)
        type_count = int(variable_type_ids.max().item()) + 1
        self.type_embedding = nn.Embedding(type_count, hidden_dim)
        self.register_buffer("variable_type_ids", variable_type_ids.long(), persistent=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool_query = nn.Parameter(torch.empty(hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.variable_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def train(self, mode: bool = True) -> "MoiraiEncoderClassifier":
        super().train(mode)
        self.moirai_tokenizer.eval()
        return self

    def forward(self, windows: torch.Tensor) -> ClassifierOutput:
        moirai_tokens = self.moirai_tokenizer(windows)
        tokens = self.input_projection(moirai_tokens)
        variable_ids = torch.arange(self.num_variables, device=tokens.device)
        metadata = self.variable_embedding(variable_ids) + self.type_embedding(self.variable_type_ids)
        tokens = self.encoder(tokens + metadata.unsqueeze(0))
        weights = torch.softmax(
            torch.einsum("bcd,d->bc", tokens, self.pool_query) / self.pool_query.numel() ** 0.5,
            dim=1,
        )
        pooled = torch.einsum("bc,bcd->bd", weights, tokens)
        logits = self.classifier(self.output_norm(pooled))
        return ClassifierOutput(
            classifier_logits=logits,
            verbalizer_logits=None,
            moirai_tokens=moirai_tokens,
            soft_tokens=tokens,
            query_hidden=pooled,
        )
