"""Prompt construction and continuous-token insertion for a causal Qwen model."""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_PREFIX = (
    "你是工业控制水处理系统的异常检测器。"
    "传感器与执行器之间应保持合理的时序和控制关系。"
    "请结合后续时间窗口的连续表示判断系统状态。\n时间窗口表示："
)
DEFAULT_SUFFIX = "\n请只在正常和异常之间进行判断。该时间窗口的系统状态为："

# Keep the legacy process prompt unchanged for comparability.  The other three
# variants are counterfactual controls: only the fixed text differs; data,
# continuous tokens, model weights, and optimization remain unchanged.
GENERIC_PREFIX = (
    "You are a time-series anomaly classifier. Determine whether the following "
    "time-window state is normal or anomalous.\nTime-window representation:\n"
)
MINIMAL_PREFIX = "Classify the following time-window state as normal or anomalous.\n"
WRONG_PROCESS_PREFIX = (
    "You monitor an online retail recommendation system. Customer preferences and "
    "item popularity should follow reasonable marketing relations. Determine whether "
    "the following time-window state is normal or anomalous.\nTime-window representation:\n"
    " Please answer carefully."
)

DTT_PROCESS_PREFIX_TEMPLATE = (
    "你正在执行WADI供水与配水工业控制系统的二分类异常检测任务。"
    "输入来自系统中的传感器和执行器，请判断该时间窗口末端的当前系统状态是正常还是异常。"
    "判断时应综合考虑单变量变化、设备当前状态，以及不同变量之间的时序和控制关系。"
    "每个连续变量后面的一个【时序Token】概括从过去到当前共{window_length}个时间步的数据；"
    "开关量和低基数变量只给出当前时刻状态，不表示其历史序列。\n"
    "以下按变量名逐项给出输入：\n"
)
DTT_SUFFIX = "\n请综合以上全部变量，只回答正常或异常。该窗口末端的系统状态为："
DTT_SYSTEM_PROMPT = (
    "你是WADI供水与配水工业控制系统的异常检测助手。"
    "你必须根据用户提供的多变量窗口信息完成二分类，只输出正常或异常。"
)
CHAT_INPUT_PLACEHOLDER = "<DTT_VARIABLE_INPUTS_7F3A9C>"


def prompt_texts(variant: str) -> tuple[str, str]:
    """Return a fixed prompt pair for a named, auditable counterfactual."""
    variants = {
        "process": (DEFAULT_PREFIX, DEFAULT_SUFFIX),
        "generic": (GENERIC_PREFIX, DEFAULT_SUFFIX),
        "minimal": (MINIMAL_PREFIX, DEFAULT_SUFFIX),
        # With the shared suffix, this prefix is also exactly 40 Qwen tokens,
        # matching the legacy process prefix.  The primary process-versus-wrong
        # comparison therefore holds prompt length constant.
        "wrong_process": (WRONG_PROCESS_PREFIX, DEFAULT_SUFFIX),
    }
    try:
        return variants[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt variant: {variant}") from exc


def dtt_prompt_texts(
    variant: str,
    window_length: int,
    numeric_mode: str = "continuous_only",
) -> tuple[str, str]:
    """Return the variable-aligned DTT task prompt.

    The process variant contains WADI-specific task context. Counterfactual
    variants retain their original prefix but still declare the input layout so
    that the aligned assembler remains well-defined.
    """
    if numeric_mode not in {"continuous_only", "all_active"}:
        raise ValueError("DTT numeric mode must be continuous_only or all_active")
    if variant == "process":
        if numeric_mode == "continuous_only":
            prefix = DTT_PROCESS_PREFIX_TEMPLATE.format(window_length=window_length)
        else:
            prefix = (
                "你正在执行WADI供水与配水工业控制系统的二分类异常检测任务。"
                "输入来自系统中的传感器和执行器，请判断该时间窗口末端的当前系统状态是正常还是异常。"
                "判断时应综合考虑单变量变化、设备当前状态，以及不同变量之间的时序和控制关系。"
                f"每个变量后面的一个【时序Token】概括从过去到当前共{window_length}个时间步的数值序列；"
                "开关量和低基数变量还会额外给出窗口末端的当前状态文本，该文本本身不表示历史。\n"
                "以下按变量名逐项给出输入：\n"
            )
        return prefix, DTT_SUFFIX
    prefix, _ = prompt_texts(variant)
    if numeric_mode == "continuous_only":
        layout = (
            f" Each continuous token summarizes {window_length} time steps through the current time;"
            " each discrete value describes only the current state. Inputs are aligned by variable name:\n"
        )
    else:
        layout = (
            f" Each variable token summarizes {window_length} numerical time steps through the current time;"
            " discrete variables additionally provide endpoint-state text. Inputs are aligned by variable name:\n"
        )
    return prefix + layout, DTT_SUFFIX


class PromptAssembler(nn.Module):
    def __init__(
        self,
        tokenizer,
        prefix: str = DEFAULT_PREFIX,
        suffix: str = DEFAULT_SUFFIX,
        *,
        use_chat_template: bool = False,
        system_prompt: str = DTT_SYSTEM_PROMPT,
    ) -> None:
        super().__init__()
        self.uses_chat_template = bool(use_chat_template)
        if use_chat_template:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prefix + CHAT_INPUT_PLACEHOLDER + suffix},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if rendered.count(CHAT_INPUT_PLACEHOLDER) != 1:
                raise ValueError("Qwen chat template did not preserve the unique DTT placeholder")
            rendered_prefix, rendered_suffix = rendered.split(CHAT_INPUT_PLACEHOLDER)
            prefix_ids = tokenizer.encode(rendered_prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(rendered_suffix, add_special_tokens=False)
        else:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if not prefix_ids or not suffix_ids:
            raise ValueError("Prompt prefix and suffix must tokenize to non-empty sequences")
        self.register_buffer("prefix_ids", torch.tensor(prefix_ids, dtype=torch.long), persistent=True)
        self.register_buffer("suffix_ids", torch.tensor(suffix_ids, dtype=torch.long), persistent=True)
        self._tokenizer = tokenizer

    def forward(
        self,
        soft_tokens: torch.Tensor,
        embedding_layer: nn.Module,
        actuator_texts: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = soft_tokens.shape[0]
        prefix_ids = self.prefix_ids.unsqueeze(0).expand(batch, -1)
        suffix_ids = self.suffix_ids.unsqueeze(0).expand(batch, -1)
        prefix = embedding_layer(prefix_ids).to(dtype=soft_tokens.dtype)
        suffix = embedding_layer(suffix_ids).to(dtype=soft_tokens.dtype)

        parts = [prefix]
        masks: list[torch.Tensor] = [torch.ones(batch, prefix.shape[1], dtype=torch.long, device=prefix.device)]

        if actuator_texts is not None:
            act_ids, act_mask = self._encode_actuator_batched(actuator_texts, device=prefix.device)
            act_emb = embedding_layer(act_ids).to(dtype=soft_tokens.dtype)
            parts.append(act_emb)
            masks.append(act_mask)

        parts.append(soft_tokens)
        masks.append(torch.ones(batch, soft_tokens.shape[1], dtype=torch.long, device=soft_tokens.device))
        parts.append(suffix)
        masks.append(torch.ones(batch, suffix.shape[1], dtype=torch.long, device=suffix.device))

        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)
        return inputs_embeds, attention_mask

    def _encode_actuator_batched(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        all_ids: list[torch.Tensor] = []
        for text in texts:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                ids = self._tokenizer.encode("无", add_special_tokens=False)
            all_ids.append(torch.tensor(ids, dtype=torch.long))
        max_len = max(len(ids) for ids in all_ids)
        pad_id = self._tokenizer.pad_token_id or 0
        padded: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded.append(torch.cat([ids, ids.new_full((pad_len,), pad_id)]))
            masks.append(torch.cat([
                torch.ones(len(ids), dtype=torch.long, device=device),
                torch.zeros(pad_len, dtype=torch.long, device=device),
            ]))
        return torch.stack(padded).to(device), torch.stack(masks).to(device)

    def forward_variable_aligned(
        self,
        soft_tokens: torch.Tensor,
        embedding_layer: nn.Module,
        *,
        active_names: tuple[str, ...],
        active_descriptions: tuple[str, ...],
        semantic_style: str,
        numeric_indices: tuple[int, ...],
        discrete_indices: tuple[int, ...],
        discrete_states: list[tuple[str, ...]],
        window_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave each variable label with its corresponding token or state."""
        batch = soft_tokens.shape[0]
        if len(discrete_states) != batch:
            raise ValueError("Each sample must provide its own discrete endpoint states")
        if soft_tokens.shape[1] != len(numeric_indices):
            raise ValueError("Numeric soft-token count does not match metadata")
        if any(len(states) != len(discrete_indices) for states in discrete_states):
            raise ValueError("Discrete state count does not match metadata")
        if len(set(numeric_indices)) != len(numeric_indices):
            raise ValueError("Numeric-token indices must be unique")
        if len(set(discrete_indices)) != len(discrete_indices):
            raise ValueError("Discrete indices must be unique")
        if any(index < 0 or index >= len(active_names) for index in (*numeric_indices, *discrete_indices)):
            raise ValueError("Variable metadata contains an out-of-range index")
        covered = set(numeric_indices) | set(discrete_indices)
        if covered != set(range(len(active_names))):
            raise ValueError("Numeric-token and discrete metadata must cover all active variables")
        if len(active_descriptions) != len(active_names):
            raise ValueError("Natural-language descriptions must match active variable names")
        if semantic_style not in {"compact", "full"}:
            raise ValueError("Semantic style must be compact or full")

        numeric_slot = {active_pos: slot for slot, active_pos in enumerate(numeric_indices)}
        discrete_slot = {active_pos: slot for slot, active_pos in enumerate(discrete_indices)}
        device = soft_tokens.device

        def text_embedding(text: str) -> torch.Tensor:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError("Variable-aligned prompt segment tokenized to an empty sequence")
            tensor = torch.tensor(ids, dtype=torch.long, device=device)
            return embedding_layer(tensor).to(dtype=soft_tokens.dtype)

        prefix = embedding_layer(self.prefix_ids.to(device)).to(dtype=soft_tokens.dtype)
        suffix = embedding_layer(self.suffix_ids.to(device)).to(dtype=soft_tokens.dtype)
        sequences: list[torch.Tensor] = []
        for sample in range(batch):
            parts = [prefix]
            for active_pos, (name, description) in enumerate(zip(active_names, active_descriptions)):
                has_numeric_token = active_pos in numeric_slot
                has_discrete_state = active_pos in discrete_slot
                if has_numeric_token:
                    if semantic_style == "compact":
                        semantic_text = f"（{description}）" if description else ""
                        header = f"{name}{semantic_text}，{window_length}步："
                    else:
                        variable_kind = "离散变量" if has_discrete_state else "连续变量"
                        if description:
                            header = (
                                f"{variable_kind}：{description}（原始变量ID：{name}），"
                                f"过去{window_length}步时序Token："
                            )
                        else:
                            header = (
                                f"{variable_kind}（原始变量ID：{name}），"
                                f"过去{window_length}步时序Token："
                            )
                    parts.append(text_embedding(header))
                    slot = numeric_slot[active_pos]
                    parts.append(soft_tokens[sample, slot : slot + 1])
                    if has_discrete_state:
                        state = discrete_states[sample][discrete_slot[active_pos]]
                        if semantic_style == "compact":
                            ending = f"，末端状态={state}\n"
                        else:
                            ending = f"，当前末端状态：{state}。\n"
                        parts.append(text_embedding(ending))
                    else:
                        parts.append(text_embedding("。\n"))
                else:
                    state = discrete_states[sample][discrete_slot[active_pos]]
                    if semantic_style == "compact":
                        semantic_text = f"（{description}）" if description else ""
                        line = f"{name}{semantic_text}={state}\n"
                    else:
                        if description:
                            line = (
                                f"离散变量：{description}（原始变量ID：{name}），"
                                f"当前状态：{state}。\n"
                            )
                        else:
                            line = f"离散变量（原始变量ID：{name}），当前状态：{state}。\n"
                    parts.append(text_embedding(line))
            parts.append(suffix)
            sequences.append(torch.cat(parts, dim=0))

        max_length = max(sequence.shape[0] for sequence in sequences)
        padded: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for sequence in sequences:
            pad_length = max_length - sequence.shape[0]
            padded.append(torch.cat([sequence.new_zeros((pad_length, sequence.shape[1])), sequence], dim=0))
            masks.append(torch.cat([
                torch.zeros(pad_length, dtype=torch.long, device=device),
                torch.ones(sequence.shape[0], dtype=torch.long, device=device),
            ]))
        return torch.stack(padded), torch.stack(masks)


def single_token_label_ids(tokenizer, labels: tuple[str, str]) -> tuple[int, int]:
    ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Verbalizer label {label!r} must be one token, got {encoded}")
        ids.append(int(encoded[0]))
    if ids[0] == ids[1]:
        raise ValueError("The two verbalizer labels map to the same token")
    return ids[0], ids[1]


def prototype_token_ids(tokenizer, words: tuple[str, ...]) -> tuple[int, ...]:
    ids: list[int] = []
    seen: set[int] = set()
    special_ids = set(tokenizer.all_special_ids)
    for word in words:
        for token_id in tokenizer.encode(word, add_special_tokens=False):
            token_id = int(token_id)
            if token_id not in seen and token_id not in special_ids:
                ids.append(token_id)
                seen.add(token_id)
    if not ids:
        raise ValueError("No usable prototype tokens were produced")
    return tuple(ids)
