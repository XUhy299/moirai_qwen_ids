"""Auditable natural-language descriptions derived from WADI tag names."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


STAGES = {
    "1": "第一工艺段",
    "2": "第二工艺段",
    "2A": "第二工艺段A支路",
    "2B": "第二工艺段B支路",
    "3": "第三工艺段",
}
COMPACT_STAGES = {
    "1": "第一段",
    "2": "第二段",
    "2A": "第二段A支路",
    "2B": "第二段B支路",
    "3": "第三段",
}
COMPACT_DEVICES = {
    "AIT": "分析仪",
    "DPIT": "差压计",
    "FIC": "流量控制器",
    "FIT": "流量计",
    "FQ": "累计流量",
    "LS": "液位开关",
    "LT": "液位计",
    "MCV": "调节阀",
    "MV": "电动阀",
    "P": "泵",
    "PIC": "压力控制器",
    "PIT": "压力计",
    "SV": "电磁阀",
}
COMPACT_SIGNALS = {
    "PV": "过程值",
    "SP": "设定值",
    "CO": "控制输出",
    "STATUS": "状态",
    "SPEED": "速度",
    "AH": "高位报警",
    "AL": "低位报警",
}
DEVICES = {
    "AIT": "分析指示变送器",
    "DPIT": "差压指示变送器",
    "FIC": "流量指示控制器",
    "FIT": "流量指示变送器",
    "FQ": "累计流量测量",
    "LS": "液位开关",
    "LT": "液位变送器",
    "MCV": "电动调节阀",
    "MV": "电动阀",
    "P": "泵",
    "PIC": "压力指示控制器",
    "PIT": "压力指示变送器",
    "SV": "电磁阀",
}
SIGNALS = {
    "PV": "过程值",
    "SP": "设定值",
    "CO": "控制器输出",
    "STATUS": "当前状态",
    "SPEED": "运行速度",
    "AH": "高液位报警状态",
    "AL": "低液位报警状态",
}
SPECIAL_VARIABLES = {
    "LEAK_DIFF_PRESSURE": "管网泄漏差压指标",
    "PLANT_START_STOP_LOG": "系统启停状态记录",
    "TOTAL_CONS_REQUIRED_FLOW": "用户侧总需求流量",
}


def describe_wadi_variable(name: str, style: str = "compact") -> str:
    """Parse a WADI tag conservatively while retaining its original identity."""
    if style not in {"compact", "full"}:
        raise ValueError("Semantic style must be compact or full")
    if name in SPECIAL_VARIABLES:
        return SPECIAL_VARIABLES[name]
    parts = name.split("_")
    if len(parts) < 4:
        return f"WADI变量{name}"
    stage, device, number = parts[0], parts[1], parts[2]
    signal = "_".join(parts[3:])
    stage_table = COMPACT_STAGES if style == "compact" else STAGES
    stage_text = stage_table.get(stage, f"{stage}工艺区域")
    if style == "compact":
        device_text = COMPACT_DEVICES.get(device, device)
        signal_text = COMPACT_SIGNALS.get(signal, signal)
        return f"{stage_text}·{device_text}·{signal_text}"
    device_text = DEVICES.get(device, f"设备类型{device}")
    signal_text = SIGNALS.get(signal, f"信号{signal}")
    return f"{stage_text}编号{number}的{device_text}（{device}）的{signal_text}"


@dataclass(frozen=True)
class VariableSemanticMap:
    rule: str
    style: str
    variables: tuple[dict[str, object], ...]

    @classmethod
    def from_names(
        cls,
        names: tuple[str, ...],
        style: str = "compact",
        *,
        variant: str = "correct",
        shuffle_seed: int = 2026,
    ) -> "VariableSemanticMap":
        if variant not in {"correct", "id_only", "shuffled"}:
            raise ValueError("Semantic variant must be correct, id_only, or shuffled")
        descriptions = tuple(describe_wadi_variable(name, style=style) for name in names)
        if variant == "correct":
            # Keep the original payload byte-for-byte compatible so historical
            # correct-semantic hashes remain meaningful.
            return cls(
                rule="由WADI标签中的工艺段、设备缩写、编号和信号后缀保守解析；Prompt同时保留原始变量ID",
                style=style,
                variables=tuple(
                    {"name": name, "description": description}
                    for name, description in zip(names, descriptions)
                ),
            )
        if variant == "id_only":
            return cls(
                rule="仅保留原始变量ID；删除变量自然语言描述，其他DTT输入保持不变",
                style=style,
                variables=tuple({"name": name, "description": ""} for name in names),
            )
        if len(names) < 2:
            raise ValueError("Shuffled semantics require at least two variables")
        cycle = list(range(len(names)))
        random.Random(shuffle_seed).shuffle(cycle)
        source_for_target = [0] * len(names)
        for position, target_index in enumerate(cycle):
            source_for_target[target_index] = cycle[(position + 1) % len(cycle)]
        if any(target == source for target, source in enumerate(source_for_target)):
            raise RuntimeError("Semantic shuffle unexpectedly retained a correct assignment")
        return cls(
            rule=(
                "使用固定全排列打乱变量自然语言描述；原始变量ID、数值Token、离散状态和变量顺序保持不变；"
                f"shuffle_seed={shuffle_seed}"
            ),
            style=style,
            variables=tuple(
                {
                    "name": name,
                    "description": descriptions[source_index],
                    "description_source_name": names[source_index],
                }
                for name, source_index in zip(names, source_for_target)
            ),
        )

    @property
    def descriptions(self) -> tuple[str, ...]:
        return tuple(str(item["description"]) for item in self.variables)

    def as_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "style": self.style, "variables": list(self.variables)}

    def sha256(self) -> str:
        canonical = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save(self, path: str | Path) -> None:
        payload = self.as_dict()
        payload["sha256"] = self.sha256()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
