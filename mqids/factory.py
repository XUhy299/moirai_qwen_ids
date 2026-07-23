"""Local-only model construction for formal experiments."""

from __future__ import annotations

import torch

from .config import ExperimentConfig
from .data import DiscreteStateVocabulary
from .model import MoiraiEncoderClassifier, MoiraiQwenClassifier
from .moirai_tokenizer import FrozenMoiraiTokenizer
from .paths import moirai_model_path, qwen_model_path, require_files
from .prompting import PromptAssembler, dtt_prompt_texts, prompt_texts, prototype_token_ids
from .projectors import DirectProjector, LinearProjector, ReprogrammingProjector


def load_qwen(config: ExperimentConfig, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = qwen_model_path(config.qwen_subdir)
    require_files(path / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    return tokenizer, model


def build_projector(
    config: ExperimentConfig,
    *,
    d_moirai: int,
    d_llm: int,
    tokenizer,
    qwen_model,
) -> torch.nn.Module:
    if config.projector == "linear":
        return LinearProjector(d_moirai, d_llm)
    if config.projector == "direct":
        return DirectProjector(
            d_moirai,
            d_llm,
            config.projector_hidden_dim,
            config.dropout,
        )
    token_ids = prototype_token_ids(tokenizer, config.prototype_words)
    with torch.no_grad():
        ids = torch.tensor(token_ids, dtype=torch.long, device=qwen_model.device)
        prototypes = qwen_model.get_input_embeddings()(ids).float().cpu()
    return ReprogrammingProjector(
        d_moirai=d_moirai,
        d_llm=d_llm,
        hidden_dim=config.projector_hidden_dim,
        attention_dim=config.attention_dim,
        n_heads=config.attention_heads,
        prototypes=prototypes,
        dropout=config.dropout,
    )


def build_model(
    config: ExperimentConfig,
    *,
    active_variable_count: int,
    variable_type_ids: torch.Tensor,
    device: torch.device,
    continuous_indices: tuple[int, ...] = (),
    discrete_indices: tuple[int, ...] = (),
    active_names: tuple[str, ...] = (),
    active_descriptions: tuple[str, ...] = (),
    discrete_vocabulary: DiscreteStateVocabulary | None = None,
) -> torch.nn.Module:
    moirai_target_dim = (
        len(continuous_indices) if config.discrete_to_text else active_variable_count
    )
    moirai = FrozenMoiraiTokenizer.from_pretrained(
        moirai_model_path(config.moirai_size),
        window_length=config.window_length,
        patch_size=config.patch_size,
        target_dim=moirai_target_dim,
        encoder_layer=config.moirai_encoder_layer,
        device=device,
    )
    d_moirai = int(moirai.forecast.module.d_model)
    if config.backbone == "none":
        if config.discrete_to_text:
            raise ValueError("The no-Qwen baseline does not support discrete-to-text mode")
        return MoiraiEncoderClassifier(
            moirai_tokenizer=moirai,
            d_moirai=d_moirai,
            variable_type_ids=variable_type_ids.to(device),
            hidden_dim=config.baseline_hidden_dim,
            layers=config.baseline_layers,
            heads=config.attention_heads,
            dropout=config.dropout,
        ).to(device)
    tokenizer, qwen = load_qwen(config, device)
    projector = build_projector(
        config,
        d_moirai=d_moirai,
        d_llm=int(qwen.config.hidden_size),
        tokenizer=tokenizer,
        qwen_model=qwen,
    ).to(device)
    if config.discrete_to_text:
        prefix, suffix = dtt_prompt_texts(config.prompt_variant, config.window_length)
    else:
        prefix, suffix = prompt_texts(config.prompt_variant)
    model = MoiraiQwenClassifier(
        moirai_tokenizer=moirai,
        qwen_causal_lm=qwen,
        tokenizer=tokenizer,
        projector=projector,
        variable_type_ids=variable_type_ids.to(device),
        labels=config.labels,
        prompt_assembler=PromptAssembler(
            tokenizer,
            prefix=prefix,
            suffix=suffix,
            use_chat_template=config.discrete_to_text,
        ),
        discrete_to_text=config.discrete_to_text,
        active_names=active_names,
        active_descriptions=active_descriptions,
        semantic_style=config.dtt_semantic_style,
        discrete_vocabulary=discrete_vocabulary,
        continuous_indices=continuous_indices,
        discrete_indices=discrete_indices,
        window_length=config.window_length,
    ).to(device)
    if config.classifier_loss_weight == 0:
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(False)
        for parameter in model.classifier_norm.parameters():
            parameter.requires_grad_(False)
    return model
