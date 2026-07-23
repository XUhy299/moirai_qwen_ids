"""Shape and gradient smoke tests, with optional local Qwen validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch import nn

from mqids.data import DiscreteStateSpec, DiscreteStateVocabulary
from mqids.losses import DualObjectiveLoss
from mqids.model import MoiraiQwenClassifier, count_trainable_parameters
from mqids.paths import qwen_model_path
from mqids.prompting import PromptAssembler, dtt_prompt_texts
from mqids.projectors import DirectProjector, LinearProjector, ReprogrammingProjector
from mqids.switch_templates import build_discrete_text


class FakeTokenizer:
    all_special_ids: list[int] = []

    def __init__(self) -> None:
        self.tokens = {"正常": 1, "异常": 2}
        self.next_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.tokens:
            return [self.tokens[text]]
        ids: list[int] = []
        for character in text:
            if character not in self.tokens:
                self.tokens[character] = self.next_id
                self.next_id += 1
            ids.append(self.tokens[character])
        return ids

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ) -> str:
        if tokenize:
            raise AssertionError("Synthetic chat smoke expects rendered text")
        rendered = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>\n"
        return rendered


class FakeMoiraiTokenizer(nn.Module):
    def __init__(self, window_length: int, d_moirai: int) -> None:
        super().__init__()
        self.projection = nn.Linear(window_length, d_moirai)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.projection(windows.transpose(1, 2)).detach()


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mix = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds, attention_mask, use_cache, return_dict):
        positions = torch.arange(
            1,
            inputs_embeds.shape[1] + 1,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).view(1, -1, 1)
        contextual = inputs_embeds.cumsum(dim=1) / positions
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(contextual)))


class FakeCausalLM(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 512) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = FakeBackbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


def build_projector(name: str, d_moirai: int, d_llm: int) -> nn.Module:
    if name == "linear":
        return LinearProjector(d_moirai, d_llm)
    if name == "direct":
        return DirectProjector(d_moirai, d_llm, hidden_dim=2 * d_moirai, dropout=0.0)
    return ReprogrammingProjector(
        d_moirai=d_moirai,
        d_llm=d_llm,
        hidden_dim=2 * d_moirai,
        attention_dim=32,
        n_heads=4,
        prototypes=torch.randn(12, d_llm),
        dropout=0.0,
    )


def run_synthetic(projector_name: str) -> None:
    torch.manual_seed(7)
    batch, window, variables = 2, 8, 6
    d_moirai, d_llm = 24, 64
    tokenizer = FakeTokenizer()
    model = MoiraiQwenClassifier(
        moirai_tokenizer=FakeMoiraiTokenizer(window, d_moirai),
        qwen_causal_lm=FakeCausalLM(d_llm),
        tokenizer=tokenizer,
        projector=build_projector(projector_name, d_moirai, d_llm),
        variable_type_ids=torch.tensor([0, 0, 1, 1, 2, 0]),
    )
    model.train()
    windows = torch.randn(batch, window, variables)
    labels = torch.tensor([0, 1])
    output = model(windows)
    losses = DualObjectiveLoss(vocab_weight=0.1)(
        output.classifier_logits,
        output.verbalizer_logits,
        labels,
    )
    losses["loss"].backward()
    projector_grads = [
        parameter.grad
        for parameter in model.projector.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if output.moirai_tokens.shape != (batch, variables, d_moirai):
        raise AssertionError(f"Bad MOIRAI shape: {output.moirai_tokens.shape}")
    if output.soft_tokens.shape != (batch, variables, d_llm):
        raise AssertionError(f"Bad soft-token shape: {output.soft_tokens.shape}")
    if output.classifier_logits.shape != (batch, 2) or output.verbalizer_logits.shape != (batch, 2):
        raise AssertionError("Both output branches must be binary logits")
    if not projector_grads or not any(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in projector_grads):
        raise AssertionError("No finite nonzero gradient reached the projector")
    if any(parameter.grad is not None for parameter in model.qwen.parameters()):
        raise AssertionError("Frozen Qwen parameters received gradients")
    if any(parameter.grad is not None for parameter in model.moirai_tokenizer.parameters()):
        raise AssertionError("Frozen MOIRAI parameters received gradients")
    print(
        f"synthetic {projector_name}: OK; loss={losses['loss'].item():.4f}; "
        f"trainable={count_trainable_parameters(model):,}"
    )


def run_synthetic_dtt(numeric_mode: str) -> None:
    """Check aligned variable/token order, endpoint states, padding and gradients."""
    torch.manual_seed(11)
    batch, window = 2, 8
    d_moirai, d_llm = 24, 64
    tokenizer = FakeTokenizer()
    vocabulary = DiscreteStateVocabulary(
        source="synthetic normal train only",
        unknown_state_rule="explicit unknown",
        variables=(
            DiscreteStateSpec(2, 2, "SWITCH_A", 1, (0.0, 1.0), (0.0, 1.0), ("关闭", "开启")),
            DiscreteStateSpec(3, 3, "SWITCH_B", 1, (0.0, 1.0), (0.0, 1.0), ("关闭", "开启")),
            DiscreteStateSpec(4, 4, "MODE", 2, (0.0, 2.0), (0.0, 2.0), ("状态0", "状态1")),
        ),
    )
    prefix, suffix = dtt_prompt_texts("process", window, numeric_mode)
    model = MoiraiQwenClassifier(
        moirai_tokenizer=FakeMoiraiTokenizer(window, d_moirai),
        qwen_causal_lm=FakeCausalLM(d_llm),
        tokenizer=tokenizer,
        projector=DirectProjector(d_moirai, d_llm, hidden_dim=48, dropout=0.0),
        variable_type_ids=torch.tensor([0, 0, 1, 1, 2, 0]),
        prompt_assembler=PromptAssembler(
            tokenizer, prefix=prefix, suffix=suffix, use_chat_template=True
        ),
        discrete_to_text=True,
        active_names=("SENSOR_A", "SENSOR_B", "SWITCH_A", "SWITCH_B", "MODE", "SENSOR_C"),
        active_descriptions=("流量传感器A", "液位传感器B", "开关A", "开关B", "模式变量", "压力传感器C"),
        dtt_numeric_mode=numeric_mode,
        discrete_vocabulary=vocabulary,
        continuous_indices=(0, 1, 5),
        discrete_indices=(2, 3, 4),
        window_length=window,
    )
    windows = torch.randn(batch, window, 6)
    windows[:, :, 2:4] = 0
    windows[0, -1, 2] = 1
    windows[1, -1, 3] = 1
    windows[:, :, 4] = 0
    windows[1, -1, 4] = 2
    endpoint_states = build_discrete_text(windows[0, :, (2, 3, 4)].T, vocabulary)
    changed_history = windows[0, :, (2, 3, 4)].T.clone()
    changed_history[:, :-1] = torch.flip(changed_history[:, :-1], dims=(1,)) + 99
    if build_discrete_text(changed_history, vocabulary) != endpoint_states:
        raise AssertionError("DTT discrete text must depend only on the endpoint state")
    output = model(windows)
    loss = DualObjectiveLoss(0.1)(
        output.classifier_logits, output.verbalizer_logits, torch.tensor([0, 1])
    )["loss"]
    loss.backward()
    expected_variables = 3 if numeric_mode == "continuous_only" else 6
    if output.moirai_tokens.shape != (batch, expected_variables, d_moirai):
        raise AssertionError(
            f"Unexpected {numeric_mode} DTT token shape: {output.moirai_tokens.shape}"
        )
    grads = [p.grad for p in model.projector.parameters() if p.grad is not None]
    if not grads or not any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
        raise AssertionError("No valid gradient crossed the variable-aligned DTT prompt")
    print(
        f"synthetic variable-aligned DTT ({numeric_mode}): "
        f"OK; loss={loss.item():.4f}"
    )


def run_real_qwen(device_name: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device_name)
    path = qwen_model_path()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    qwen = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model = MoiraiQwenClassifier(
        moirai_tokenizer=FakeMoiraiTokenizer(8, 24),
        qwen_causal_lm=qwen,
        tokenizer=tokenizer,
        projector=DirectProjector(24, qwen.config.hidden_size, hidden_dim=48, dropout=0.0),
        variable_type_ids=torch.tensor([0, 1, 0, 2], device=device),
    ).to(device)
    model.train()
    output = model(torch.randn(1, 8, 4, device=device))
    loss = DualObjectiveLoss(0.1)(output.classifier_logits, output.verbalizer_logits, torch.tensor([1], device=device))["loss"]
    loss.backward()
    grad = next(parameter.grad for parameter in model.projector.parameters() if parameter.grad is not None)
    if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
        raise AssertionError("Real Qwen did not pass a valid gradient to the projector")
    if any(parameter.grad is not None for parameter in model.qwen.parameters()):
        raise AssertionError("Frozen real Qwen received parameter gradients")
    print(f"real Qwen: OK; device={device}; loss={loss.item():.4f}")

    vocabulary = DiscreteStateVocabulary(
        source="synthetic normal train only",
        unknown_state_rule="explicit unknown",
        variables=(
            DiscreteStateSpec(1, 1, "PUMP_STATUS", 1, (0.0, 1.0), (0.0, 1.0), ("停止", "运行")),
            DiscreteStateSpec(3, 3, "VALVE_STATUS", 1, (0.0, 1.0), (0.0, 1.0), ("关闭", "开启")),
        ),
    )
    dtt_model = MoiraiQwenClassifier(
        moirai_tokenizer=FakeMoiraiTokenizer(8, 24),
        qwen_causal_lm=qwen,
        tokenizer=tokenizer,
        projector=DirectProjector(24, qwen.config.hidden_size, hidden_dim=48, dropout=0.0),
        variable_type_ids=torch.tensor([0, 1, 0, 1], device=device),
        prompt_assembler=PromptAssembler(
            tokenizer,
            prefix=dtt_prompt_texts("process", 8)[0],
            suffix=dtt_prompt_texts("process", 8)[1],
            use_chat_template=True,
        ),
        discrete_to_text=True,
        active_names=("FLOW_A", "PUMP_STATUS", "LEVEL_B", "VALVE_STATUS"),
        active_descriptions=("流量传感器A", "泵状态", "液位传感器B", "阀门状态"),
        discrete_vocabulary=vocabulary,
        continuous_indices=(0, 2),
        discrete_indices=(1, 3),
        window_length=8,
    ).to(device)
    dtt_windows = torch.randn(2, 8, 4, device=device)
    dtt_windows[:, :, 1] = 0
    dtt_windows[1, -1, 1] = 1
    dtt_windows[:, :, 3] = 1
    dtt_output = dtt_model(dtt_windows)
    dtt_loss = DualObjectiveLoss(0.1)(
        dtt_output.classifier_logits,
        dtt_output.verbalizer_logits,
        torch.tensor([0, 1], device=device),
    )["loss"]
    dtt_loss.backward()
    dtt_grads = [p.grad for p in dtt_model.projector.parameters() if p.grad is not None]
    if not dtt_grads or not any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in dtt_grads):
        raise AssertionError("Real Qwen did not pass a valid gradient through aligned DTT")
    print(f"real Qwen variable-aligned DTT: OK; device={device}; loss={dtt_loss.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-qwen", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    for name in ("linear", "direct", "reprogramming"):
        run_synthetic(name)
    for numeric_mode in ("continuous_only", "all_active"):
        run_synthetic_dtt(numeric_mode)
    if args.real_qwen:
        run_real_qwen(args.device)


if __name__ == "__main__":
    main()
