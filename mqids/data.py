"""Leakage-aware WADI metadata, split and sliding-window utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChannelMetadata:
    names: tuple[str, ...]
    active_indices: tuple[int, ...]
    constant_indices: tuple[int, ...]
    type_ids: tuple[int, ...]
    type_names: tuple[str, ...] = ("continuous", "binary", "low_cardinality")

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(self.names[index] for index in self.active_indices)

    @property
    def continuous_indices(self) -> tuple[int, ...]:
        """Positional indices (within the active-variable subset) of continuous (type 0) variables."""
        return tuple(
            i
            for i, tid in enumerate(self.type_ids)
            if tid == 0
        )

    @property
    def discrete_indices(self) -> tuple[int, ...]:
        """Positional indices (within the active-variable subset) of binary (type 1) and low-cardinality (type 2) variables."""
        return tuple(
            i
            for i, tid in enumerate(self.type_ids)
            if tid != 0
        )

    @property
    def discrete_type_ids(self) -> tuple[int, ...]:
        """Type IDs for only the discrete variables, in the same order as discrete_indices."""
        return tuple(tid for tid in self.type_ids if tid != 0)

    @property
    def discrete_names(self) -> tuple[str, ...]:
        """Sensor names for discrete variables only (looked up via original 110-dim index)."""
        return tuple(
            self.names[self.active_indices[i]]
            for i in self.discrete_indices
        )

    def save(self, path: str | Path) -> None:
        output = asdict(self)
        output["active_names"] = list(self.active_names)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class DiscreteStateSpec:
    """A train-derived, globally stable state vocabulary for one variable."""

    active_position: int
    original_index: int
    name: str
    type_id: int
    values: tuple[float, ...]
    raw_values: tuple[float, ...]
    state_names: tuple[str, ...]
    scaler_mean: float = 0.0
    scaler_scale: float = 1.0


@dataclass(frozen=True)
class DiscreteStateVocabulary:
    source: str
    unknown_state_rule: str
    variables: tuple[DiscreteStateSpec, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["variables"] = [asdict(item) for item in self.variables]
        return payload

    def sha256(self) -> str:
        canonical = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save(self, path: str | Path) -> None:
        output = self.as_dict()
        output["sha256"] = self.sha256()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class AttackSupportSplit:
    attack_start: int
    attack_end_exclusive: int
    support_endpoints: np.ndarray
    query_endpoints: np.ndarray
    excluded_start: int
    excluded_end_exclusive: int


def load_sensor_names(path: str | Path, expected_channels: int) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    )
    if len(names) != expected_channels:
        raise ValueError(f"Expected {expected_channels} sensor names, found {len(names)}")
    return names


def _stream_min_max(array: np.ndarray, chunk_size: int = 65_536) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(array.shape[1], np.inf, dtype=np.float64)
    maxs = np.full(array.shape[1], -np.inf, dtype=np.float64)
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size], dtype=np.float64)
        mins = np.minimum(mins, np.nanmin(chunk, axis=0))
        maxs = np.maximum(maxs, np.nanmax(chunk, axis=0))
    return mins, maxs


def infer_channel_metadata(
    train_x: np.ndarray,
    names: Sequence[str],
    *,
    constant_tolerance: float = 1e-8,
    cardinality_sample_size: int = 200_000,
) -> ChannelMetadata:
    if train_x.ndim != 2:
        raise ValueError(f"Expected [time, channels], got {train_x.shape}")
    if len(names) != train_x.shape[1]:
        raise ValueError("Sensor-name count does not match the WADI array")

    # Channel selection is deliberately train-only.  Do not add an escape hatch
    # for validation/test arrays here: that caused the pre-repair DTT leakage.
    mins, maxs = _stream_min_max(train_x)

    constant_mask = np.isclose(mins, maxs, rtol=0.0, atol=constant_tolerance)
    active = np.flatnonzero(~constant_mask)

    sample_count = min(len(train_x), cardinality_sample_size)
    sample_rows = np.linspace(0, len(train_x) - 1, sample_count, dtype=np.int64)
    sample = np.asarray(train_x[sample_rows], dtype=np.float64)
    type_ids: list[int] = []
    for index in active:
        unique_count = np.unique(sample[:, index]).size
        if unique_count <= 2:
            type_ids.append(1)
        elif unique_count <= 16:
            type_ids.append(2)
        else:
            type_ids.append(0)
    return ChannelMetadata(
        names=tuple(names),
        active_indices=tuple(int(index) for index in active),
        constant_indices=tuple(int(index) for index in np.flatnonzero(constant_mask)),
        type_ids=tuple(type_ids),
    )


def _column_unique_values(array: np.ndarray, index: int, chunk_size: int = 65_536) -> np.ndarray:
    values = np.empty(0, dtype=np.float64)
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size, index], dtype=np.float64)
        chunk = chunk[np.isfinite(chunk)]
        values = np.union1d(values, np.unique(chunk))
    return values


def infer_discrete_state_vocabulary(
    train_x: np.ndarray,
    metadata: ChannelMetadata,
    *,
    source: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> DiscreteStateVocabulary:
    """Build fixed per-variable states using normal training data only."""
    variables: list[DiscreteStateSpec] = []
    if len(scaler_mean) != train_x.shape[1] or len(scaler_scale) != train_x.shape[1]:
        raise ValueError("Scaler metadata does not match WADI channel count")
    for active_position in metadata.discrete_indices:
        original_index = metadata.active_indices[active_position]
        type_id = metadata.type_ids[active_position]
        values = _column_unique_values(train_x, original_index)
        name = metadata.names[original_index]
        if values.size == 0:
            raise ValueError(f"Discrete variable {name!r} has no finite train values")
        if type_id == 1 and values.size != 2:
            raise ValueError(f"Binary variable {name!r} has {values.size} train states")
        if values.size > 16:
            raise ValueError(f"Low-cardinality variable {name!r} has {values.size} train states")
        raw_values = values * float(scaler_scale[original_index]) + float(scaler_mean[original_index])
        rounded_raw = np.rint(raw_values)
        if not np.allclose(raw_values, rounded_raw, rtol=0.0, atol=1e-6):
            state_names = tuple(f"状态{i}（原始值={value:g}）" for i, value in enumerate(raw_values))
        elif "_MV_" in name and tuple(rounded_raw.astype(int)) == (0, 1, 2):
            state_names = ("过渡中", "关闭", "开启")
        elif "_P_" in name and name.endswith("_STATUS") and tuple(rounded_raw.astype(int)) == (1, 2):
            state_names = ("停止", "运行")
        elif "_LS_" in name and tuple(rounded_raw.astype(int)) == (0, 1):
            state_names = ("未触发", "触发")
        else:
            state_names = tuple(f"状态编码{int(value)}" for value in rounded_raw)
        variables.append(
            DiscreteStateSpec(
                active_position=int(active_position),
                original_index=int(original_index),
                name=name,
                type_id=int(type_id),
                values=tuple(float(value) for value in values),
                raw_values=tuple(float(value) for value in raw_values),
                state_names=state_names,
                scaler_mean=float(scaler_mean[original_index]),
                scaler_scale=float(scaler_scale[original_index]),
            )
        )
    return DiscreteStateVocabulary(
        source=source,
        unknown_state_rule=(
            "训练期词表中无匹配值时，使用训练期scaler还原并输出未知状态的原始值；"
            "不得按窗口重新编号"
        ),
        variables=tuple(variables),
    )


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def split_single_attack_support_query(
    labels: np.ndarray,
    *,
    window_length: int,
    support_fraction: float,
    guard: int,
) -> AttackSupportSplit:
    """Split one attack chronologically; this is not event-independent validation."""
    labels = np.asarray(labels).reshape(-1)
    runs = contiguous_true_runs(labels == 1)
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one validation attack run, found {len(runs)}")
    attack_start, attack_end = runs[0]
    guard = max(int(guard), int(window_length))
    raw_cut = attack_start + int((attack_end - attack_start) * support_fraction)
    left_guard = guard // 2
    right_guard = guard - left_guard
    support_stop = raw_cut - left_guard
    query_start = raw_cut + right_guard
    if support_stop <= attack_start or query_start >= attack_end:
        raise ValueError("Attack event is too short for the requested fraction and guard")
    return AttackSupportSplit(
        attack_start=attack_start,
        attack_end_exclusive=attack_end,
        support_endpoints=np.arange(attack_start, support_stop, dtype=np.int64),
        query_endpoints=np.arange(query_start, attack_end, dtype=np.int64),
        excluded_start=support_stop,
        excluded_end_exclusive=query_start,
    )


def sample_endpoints(
    candidates: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if count >= len(candidates):
        return np.sort(candidates)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=count, replace=False))


class WadiWindowDataset(Dataset):
    """Return `[window, active_variables]` and endpoint labels."""

    def __init__(
        self,
        x: np.ndarray,
        endpoints: Sequence[int],
        window_length: int,
        active_indices: Sequence[int],
        labels: np.ndarray | None = None,
        fixed_label: int | None = None,
    ) -> None:
        if labels is None and fixed_label is None:
            raise ValueError("Either labels or fixed_label is required")
        self.x = x
        self.endpoints = np.asarray(endpoints, dtype=np.int64)
        self.window_length = int(window_length)
        self.active_indices = np.asarray(active_indices, dtype=np.int64)
        self.labels = labels
        self.fixed_label = fixed_label
        if self.endpoints.size and self.endpoints.min() < self.window_length - 1:
            raise ValueError("An endpoint does not have enough causal history")
        if self.endpoints.size and self.endpoints.max() >= len(x):
            raise ValueError("An endpoint lies outside the feature array")
        if labels is not None and len(labels) != len(x):
            raise ValueError("Feature and label lengths differ")

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        endpoint = int(self.endpoints[index])
        start = endpoint - self.window_length + 1
        window = np.asarray(self.x[start : endpoint + 1, self.active_indices], dtype=np.float32)
        label = int(self.fixed_label if self.labels is None else self.labels[endpoint])
        return {
            "window": torch.from_numpy(np.ascontiguousarray(window)),
            "label": torch.tensor(label, dtype=torch.long),
            "endpoint": torch.tensor(endpoint, dtype=torch.long),
        }


class SyntheticWindowDataset(Dataset):
    """Read a validated offline synthetic-anomaly package."""

    REQUIRED_FILES = (
        "synthetic_windows.npy",
        "point_masks.npy",
        "channel_masks.npy",
        "endpoint_labels.npy",
        "source_endpoints.npy",
        "operator_codes.npy",
        "channel_metadata.json",
        "dataset_summary.json",
    )

    def __init__(
        self,
        package_dir: str | Path,
        *,
        expected_window_length: int,
        expected_active_indices: Sequence[int],
        expected_type_ids: Sequence[int],
        expected_train_x_sha256: str,
        expected_train_length: int,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        missing = [
            str(self.package_dir / name)
            for name in self.REQUIRED_FILES
            if not (self.package_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Synthetic anomaly package is incomplete:\n" + "\n".join(missing)
            )
        with (self.package_dir / "dataset_summary.json").open("r", encoding="utf-8") as handle:
            self.summary = json.load(handle)
        with (self.package_dir / "channel_metadata.json").open("r", encoding="utf-8") as handle:
            package_metadata = json.load(handle)

        if self.summary.get("method") != "full_five_type_mixture":
            raise ValueError("Synthetic package is not a Full five-type mixture")
        if self.summary.get("validation_or_test_arrays_opened") is not False:
            raise ValueError("Synthetic package does not attest strict train-only generation")
        if self.summary.get("input_x_sha256") != expected_train_x_sha256:
            raise ValueError("Synthetic package was not generated from the current normal train X")
        if tuple(package_metadata.get("active_indices", ())) != tuple(expected_active_indices):
            raise ValueError("Synthetic active-channel order differs from the current training protocol")
        if tuple(package_metadata.get("type_ids", ())) != tuple(expected_type_ids):
            raise ValueError("Synthetic channel types differ from the current training protocol")

        self.windows = np.load(self.package_dir / "synthetic_windows.npy", mmap_mode="r")
        self.point_masks = np.load(self.package_dir / "point_masks.npy", mmap_mode="r")
        self.channel_masks = np.load(self.package_dir / "channel_masks.npy", mmap_mode="r")
        self.labels = np.load(self.package_dir / "endpoint_labels.npy", mmap_mode="r")
        self.endpoints = np.load(self.package_dir / "source_endpoints.npy", mmap_mode="r")
        self.operator_codes = np.load(self.package_dir / "operator_codes.npy", mmap_mode="r")

        expected_shape = (
            len(self.windows),
            int(expected_window_length),
            len(tuple(expected_active_indices)),
        )
        if self.windows.shape != expected_shape:
            raise ValueError(
                f"Synthetic windows have shape {self.windows.shape}, expected {expected_shape}"
            )
        if self.windows.dtype != np.float32:
            raise ValueError(f"Synthetic windows must be float32, got {self.windows.dtype}")
        if self.point_masks.shape != expected_shape[:2]:
            raise ValueError("Synthetic point-mask shape does not match windows")
        if self.channel_masks.shape != (expected_shape[0], expected_shape[2]):
            raise ValueError("Synthetic channel-mask shape does not match windows")
        for name, array in (
            ("endpoint_labels", self.labels),
            ("source_endpoints", self.endpoints),
            ("operator_codes", self.operator_codes),
        ):
            if array.shape != (expected_shape[0],):
                raise ValueError(f"Synthetic {name} shape does not match windows")
        if not np.all(self.point_masks[:, -1]):
            raise ValueError("A synthetic sample is not anomalous at the endpoint")
        if not np.all(self.channel_masks.any(axis=1)):
            raise ValueError("A synthetic sample has no changed channel")
        if not np.all(self.labels == 1):
            raise ValueError("Synthetic endpoint labels must all be one")
        if len(self.endpoints) and (
            int(self.endpoints.min()) < expected_window_length - 1
            or int(self.endpoints.max()) >= expected_train_length
        ):
            raise ValueError("A synthetic source endpoint lies outside normal train X")
        for start in range(0, len(self.windows), 512):
            if not np.all(np.isfinite(self.windows[start : start + 512])):
                raise ValueError("Synthetic windows contain NaN or infinite values")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = np.asarray(self.windows[index], dtype=np.float32).copy()
        return {
            "window": torch.from_numpy(window),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "endpoint": torch.tensor(int(self.endpoints[index]), dtype=torch.long),
        }

    def close(self) -> None:
        for name in (
            "windows",
            "point_masks",
            "channel_masks",
            "labels",
            "endpoints",
            "operator_codes",
        ):
            array = getattr(self, name, None)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()

    def __del__(self) -> None:
        self.close()


def build_stratified_epoch_schedules(
    operator_codes: Sequence[int] | np.ndarray,
    *,
    samples_per_epoch: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Build deterministic, non-overlapping per-epoch synthetic selections.

    Quotas preserve the operator proportions in the full offline package using
    largest-remainder allocation. An operator pool is never reused across
    epochs; fail closed when the package cannot support the requested schedule.
    """

    codes = np.asarray(operator_codes, dtype=np.int64)
    if codes.ndim != 1 or codes.size == 0:
        raise ValueError("operator_codes must be a non-empty one-dimensional array")
    if samples_per_epoch < 1 or epochs < 1:
        raise ValueError("samples_per_epoch and epochs must be positive")
    if samples_per_epoch * epochs > len(codes):
        raise ValueError("Synthetic package is too small for a non-overlapping epoch schedule")

    unique_codes, counts = np.unique(codes, return_counts=True)
    exact_quotas = counts.astype(np.float64) / counts.sum() * samples_per_epoch
    quotas = np.floor(exact_quotas).astype(np.int64)
    remaining = samples_per_epoch - int(quotas.sum())
    fractional = exact_quotas - quotas
    order = np.lexsort((unique_codes, -fractional))
    quotas[order[:remaining]] += 1

    required = quotas * epochs
    insufficient = [
        (int(code), int(need), int(available))
        for code, need, available in zip(unique_codes, required, counts)
        if need > available
    ]
    if insufficient:
        raise ValueError(
            "Synthetic operator pools cannot support non-overlapping stratified rotation: "
            f"{insufficient}"
        )

    rng = np.random.default_rng(seed)
    shuffled_by_code = {
        int(code): rng.permutation(np.flatnonzero(codes == code)).astype(np.int64)
        for code in unique_codes
    }
    schedules: list[np.ndarray] = []
    for epoch_index in range(epochs):
        parts = []
        for code, quota in zip(unique_codes, quotas):
            start = epoch_index * int(quota)
            stop = start + int(quota)
            parts.append(shuffled_by_code[int(code)][start:stop])
        selected = np.concatenate(parts).astype(np.int64, copy=False)
        selected = selected[rng.permutation(len(selected))]
        schedules.append(selected)

    combined = np.concatenate(schedules)
    if len(np.unique(combined)) != len(combined):
        raise AssertionError("Epoch-stratified synthetic schedules unexpectedly overlap")
    return tuple(schedules)


class EpochRotatingSyntheticDataset(Dataset):
    """Expose one deterministic synthetic subset per training epoch."""

    def __init__(
        self,
        dataset: SyntheticWindowDataset,
        schedules: Sequence[Sequence[int] | np.ndarray],
    ) -> None:
        if not schedules:
            raise ValueError("At least one synthetic epoch schedule is required")
        normalized = tuple(np.asarray(schedule, dtype=np.int64) for schedule in schedules)
        expected_length = len(normalized[0])
        if expected_length < 1 or any(len(schedule) != expected_length for schedule in normalized):
            raise ValueError("All synthetic epoch schedules must have the same positive length")
        for schedule in normalized:
            if schedule.ndim != 1:
                raise ValueError("Synthetic epoch schedules must be one-dimensional")
            if int(schedule.min()) < 0 or int(schedule.max()) >= len(dataset):
                raise ValueError("Synthetic epoch schedule contains an out-of-range index")
        self.dataset = dataset
        self.schedules = normalized
        self.epoch_index = 0

    @property
    def indices(self) -> np.ndarray:
        return self.schedules[self.epoch_index]

    def set_epoch(self, epoch_index: int) -> None:
        if epoch_index < 0 or epoch_index >= len(self.schedules):
            raise ValueError(f"Synthetic epoch index {epoch_index} is out of range")
        self.epoch_index = int(epoch_index)

    def __len__(self) -> int:
        return len(self.schedules[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.dataset[int(self.indices[index])]
