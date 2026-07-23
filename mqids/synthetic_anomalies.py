"""Dataset-agnostic synthetic anomalies for multivariate time-series windows.

The generator is deliberately fitted on normal training data only.  It emits
active-channel windows, point/channel masks, and enough metadata to reproduce
each mutation.  No validation or test array is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


OPERATOR_NAMES = (
    "spike",
    "shift_ramp",
    "flatline",
    "soft_patch_replacement",
    "dependency_break",
)


@dataclass(frozen=True)
class SyntheticAnomalyConfig:
    operator_probabilities: tuple[float, ...] = (0.15, 0.25, 0.15, 0.30, 0.15)
    max_segment_ratio: float = 0.40
    spike_min_severity: float = 1.5
    spike_max_severity: float = 4.0
    shift_min_severity: float = 0.5
    shift_max_severity: float = 2.5
    ramp_probability: float = 0.5
    replacement_min_alpha: float = 0.3
    replacement_max_alpha: float = 1.0
    single_channel_probability: float = 0.60
    small_group_probability: float = 0.30
    broad_group_probability: float = 0.10
    max_absolute_robust_z: float = 8.0
    max_generation_attempts: int = 32

    def validate(self) -> None:
        probabilities = np.asarray(self.operator_probabilities, dtype=np.float64)
        if probabilities.shape != (len(OPERATOR_NAMES),):
            raise ValueError(f"Expected {len(OPERATOR_NAMES)} operator probabilities")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Operator probabilities must be non-negative and sum to one")
        channel_sum = (
            self.single_channel_probability
            + self.small_group_probability
            + self.broad_group_probability
        )
        if not np.isclose(channel_sum, 1.0):
            raise ValueError("Channel-count probabilities must sum to one")
        if not 0 < self.max_segment_ratio <= 1:
            raise ValueError("max_segment_ratio must lie in (0, 1]")
        if not 0 <= self.ramp_probability <= 1:
            raise ValueError("ramp_probability must lie in [0, 1]")
        if not 0 < self.replacement_min_alpha <= self.replacement_max_alpha <= 1:
            raise ValueError("Replacement alpha range must lie in (0, 1]")

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["operator_names"] = list(OPERATOR_NAMES)
        return payload


@dataclass(frozen=True)
class SyntheticTrainingStatistics:
    active_indices: tuple[int, ...]
    type_ids: tuple[int, ...]
    medians: np.ndarray
    robust_scales: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    discrete_values: tuple[tuple[float, ...], ...]

    @property
    def continuous_positions(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.type_ids) == 0)

    @property
    def discrete_positions(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.type_ids) != 0)

    def as_dict(self) -> dict[str, object]:
        return {
            "active_indices": list(self.active_indices),
            "type_ids": list(self.type_ids),
            "medians": self.medians.tolist(),
            "robust_scales": self.robust_scales.tolist(),
            "lower_bounds": self.lower_bounds.tolist(),
            "upper_bounds": self.upper_bounds.tolist(),
            "discrete_values": [list(values) for values in self.discrete_values],
        }


@dataclass(frozen=True)
class SyntheticSample:
    window: np.ndarray
    point_mask: np.ndarray
    channel_mask: np.ndarray
    metadata: dict[str, object]


def _finite_unique_values(
    source: np.ndarray,
    original_index: int,
    *,
    chunk_size: int = 65_536,
) -> tuple[float, ...]:
    values = np.empty(0, dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        chunk = np.asarray(source[start : start + chunk_size, original_index], dtype=np.float64)
        values = np.union1d(values, np.unique(chunk[np.isfinite(chunk)]))
    return tuple(float(value) for value in values)


def fit_synthetic_statistics(
    normal_train_x: np.ndarray,
    active_indices: Sequence[int],
    type_ids: Sequence[int],
    *,
    sample_size: int = 100_000,
    max_absolute_robust_z: float = 8.0,
) -> SyntheticTrainingStatistics:
    """Fit scale, bounds, and legal discrete states from normal training X."""
    if normal_train_x.ndim != 2:
        raise ValueError(f"Expected [time, channels], got {normal_train_x.shape}")
    active = np.asarray(active_indices, dtype=np.int64)
    types = np.asarray(type_ids, dtype=np.int64)
    if active.size == 0 or active.size != types.size:
        raise ValueError("active_indices and type_ids must be non-empty and aligned")
    if active.min() < 0 or active.max() >= normal_train_x.shape[1]:
        raise ValueError("An active channel index lies outside the training array")

    count = min(int(sample_size), len(normal_train_x))
    rows = np.linspace(0, len(normal_train_x) - 1, count, dtype=np.int64)
    sampled = np.asarray(normal_train_x[np.ix_(rows, active)], dtype=np.float64)
    if not np.all(np.isfinite(sampled)):
        raise ValueError("Normal training sample contains NaN or infinite values")

    medians = np.median(sampled, axis=0)
    mad = np.median(np.abs(sampled - medians), axis=0)
    q001, q25, q75, q999 = np.quantile(sampled, (0.001, 0.25, 0.75, 0.999), axis=0)
    robust_scales = 1.4826 * mad
    iqr_scales = (q75 - q25) / 1.349
    std_scales = np.std(sampled, axis=0)
    robust_scales = np.where(robust_scales > 1e-8, robust_scales, iqr_scales)
    robust_scales = np.where(robust_scales > 1e-8, robust_scales, std_scales)
    robust_scales = np.where(robust_scales > 1e-8, robust_scales, 1.0)

    # Permit anomalies beyond normal quantiles while preventing numerical explosions.
    lower_bounds = np.maximum(
        np.min(sampled, axis=0) - 3.0 * robust_scales,
        medians - max_absolute_robust_z * robust_scales,
    )
    upper_bounds = np.minimum(
        np.max(sampled, axis=0) + 3.0 * robust_scales,
        medians + max_absolute_robust_z * robust_scales,
    )

    discrete_values: list[tuple[float, ...]] = []
    for position, (original_index, type_id) in enumerate(zip(active, types, strict=True)):
        if type_id == 0:
            discrete_values.append(())
            continue
        values = _finite_unique_values(normal_train_x, int(original_index))
        if not values:
            raise ValueError(f"Discrete channel at active position {position} has no values")
        if len(values) > 20:
            raise ValueError(
                f"Discrete channel at active position {position} has {len(values)} states"
            )
        discrete_values.append(values)

    return SyntheticTrainingStatistics(
        active_indices=tuple(int(index) for index in active),
        type_ids=tuple(int(type_id) for type_id in types),
        medians=medians,
        robust_scales=robust_scales,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        discrete_values=tuple(discrete_values),
    )


class FullSyntheticAnomalyGenerator:
    """Generate a mixture of five common multivariate time-series anomalies."""

    def __init__(
        self,
        normal_train_x: np.ndarray,
        statistics: SyntheticTrainingStatistics,
        *,
        window_length: int,
        config: SyntheticAnomalyConfig | None = None,
    ) -> None:
        self.source = normal_train_x
        self.statistics = statistics
        self.window_length = int(window_length)
        self.config = config or SyntheticAnomalyConfig()
        self.config.validate()
        self.active_indices = np.asarray(statistics.active_indices, dtype=np.int64)
        if self.window_length < 4:
            raise ValueError("window_length must be at least four")
        if len(normal_train_x) < 2 * self.window_length:
            raise ValueError("Normal training data is too short for donor-window generation")
        if len(statistics.continuous_positions) == 0:
            raise ValueError("At least one continuous channel is required")

    def _read_window(self, endpoint: int) -> np.ndarray:
        start = int(endpoint) - self.window_length + 1
        if start < 0 or endpoint >= len(self.source):
            raise ValueError(f"Endpoint {endpoint} does not have a valid causal window")
        return np.asarray(
            self.source[start : endpoint + 1, self.active_indices], dtype=np.float64
        ).copy()

    def _choose_channels(
        self,
        rng: np.random.Generator,
        pool: np.ndarray,
        *,
        allow_broad: bool = True,
    ) -> np.ndarray:
        if pool.size == 0:
            raise ValueError("Cannot sample from an empty channel pool")
        draw = rng.random()
        if draw < self.config.single_channel_probability or pool.size == 1:
            count = 1
        elif draw < self.config.single_channel_probability + self.config.small_group_probability:
            count = int(rng.integers(2, min(4, pool.size) + 1))
        else:
            count = max(2, int(round(np.sqrt(pool.size)))) if allow_broad else min(4, pool.size)
        count = min(count, pool.size)
        return np.sort(rng.choice(pool, size=count, replace=False).astype(np.int64))

    def _segment(self, rng: np.random.Generator, *, point_like: bool) -> tuple[int, int]:
        if point_like:
            duration = int(rng.integers(1, min(3, self.window_length) + 1))
        else:
            minimum = max(2, int(np.ceil(0.05 * self.window_length)))
            maximum = max(minimum, int(np.floor(self.config.max_segment_ratio * self.window_length)))
            duration = int(rng.integers(minimum, maximum + 1))
        return self.window_length - duration, self.window_length

    def _donor_patch(
        self,
        rng: np.random.Generator,
        *,
        duration: int,
        source_endpoint: int,
    ) -> tuple[np.ndarray, int, int]:
        source_start = source_endpoint - self.window_length + 1
        for _ in range(64):
            donor_start = int(rng.integers(0, len(self.source) - duration + 1))
            donor_end = donor_start + duration
            if donor_end <= source_start or donor_start > source_endpoint:
                patch = np.asarray(
                    self.source[donor_start:donor_end, self.active_indices], dtype=np.float64
                )
                return patch, donor_start, donor_end - 1
        raise RuntimeError("Could not sample a non-overlapping donor patch")

    def _clip_continuous(self, window: np.ndarray, channels: np.ndarray) -> None:
        continuous = channels[np.asarray(self.statistics.type_ids)[channels] == 0]
        if continuous.size:
            window[:, continuous] = np.clip(
                window[:, continuous],
                self.statistics.lower_bounds[continuous],
                self.statistics.upper_bounds[continuous],
            )

    def _apply_operator(
        self,
        clean: np.ndarray,
        source_endpoint: int,
        operator: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, object]]:
        output = clean.copy()
        continuous = self.statistics.continuous_positions
        all_channels = np.arange(clean.shape[1], dtype=np.int64)
        donor_start: int | None = None
        donor_endpoint: int | None = None
        alpha: float | None = None
        severity: float | None = None
        variant: str | None = None

        if operator == "spike":
            start, end = self._segment(rng, point_like=True)
            channels = self._choose_channels(rng, continuous)
            severity = float(rng.uniform(self.config.spike_min_severity, self.config.spike_max_severity))
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(channels))
            output[start:end, channels] += signs * severity * self.statistics.robust_scales[channels]
        elif operator == "shift_ramp":
            start, end = self._segment(rng, point_like=False)
            channels = self._choose_channels(rng, continuous)
            severity = float(rng.uniform(self.config.shift_min_severity, self.config.shift_max_severity))
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(channels))
            delta = signs * severity * self.statistics.robust_scales[channels]
            if rng.random() < self.config.ramp_probability:
                variant = "ramp"
                weights = np.linspace(1.0 / (end - start), 1.0, end - start)[:, None]
                output[start:end, channels] += weights * delta
            else:
                variant = "level_shift"
                output[start:end, channels] += delta
        elif operator == "flatline":
            start, end = self._segment(rng, point_like=False)
            channels = self._choose_channels(rng, continuous)
            output[start:end, channels] = output[start - 1, channels]
        elif operator in {"soft_patch_replacement", "dependency_break"}:
            start, end = self._segment(rng, point_like=False)
            channels = self._choose_channels(
                rng, all_channels, allow_broad=operator == "soft_patch_replacement"
            )
            donor, donor_start, donor_endpoint = self._donor_patch(
                rng, duration=end - start, source_endpoint=source_endpoint
            )
            if operator == "soft_patch_replacement":
                alpha = float(
                    rng.uniform(
                        self.config.replacement_min_alpha,
                        self.config.replacement_max_alpha,
                    )
                )
                selected_types = np.asarray(self.statistics.type_ids)[channels]
                continuous_selected = channels[selected_types == 0]
                discrete_selected = channels[selected_types != 0]
                if continuous_selected.size:
                    output[start:end, continuous_selected] = (
                        (1.0 - alpha) * output[start:end, continuous_selected]
                        + alpha * donor[:, continuous_selected]
                    )
                if discrete_selected.size:
                    output[start:end, discrete_selected] = donor[:, discrete_selected]
            else:
                output[start:end, channels] = donor[:, channels]
        else:
            raise ValueError(f"Unknown synthetic anomaly operator: {operator!r}")

        self._clip_continuous(output, channels)
        metadata: dict[str, object] = {
            "operator": operator,
            "variant": variant,
            "channels": [int(channel) for channel in channels],
            "start": int(start),
            "end_exclusive": int(end),
            "severity": severity,
            "alpha": alpha,
            "donor_start": donor_start,
            "donor_endpoint": donor_endpoint,
        }
        return output, metadata

    def generate(
        self,
        source_endpoint: int,
        rng: np.random.Generator,
        *,
        operator: str | None = None,
    ) -> SyntheticSample:
        clean = self._read_window(source_endpoint)
        if not np.all(np.isfinite(clean)):
            raise ValueError(f"Source window ending at {source_endpoint} is not finite")
        probabilities = np.asarray(self.config.operator_probabilities, dtype=np.float64)
        selected = operator or str(rng.choice(OPERATOR_NAMES, p=probabilities))
        if selected not in OPERATOR_NAMES:
            raise ValueError(f"Unknown synthetic anomaly operator: {selected!r}")
        for attempt in range(1, self.config.max_generation_attempts + 1):
            output, metadata = self._apply_operator(clean, source_endpoint, selected, rng)
            changed = ~np.isclose(output, clean, rtol=0.0, atol=1e-7)
            point_mask = np.any(changed, axis=1)
            channel_mask = np.any(changed, axis=0)
            if np.all(np.isfinite(output)) and point_mask[-1] and channel_mask.any():
                metadata.update(
                    {
                        "source_endpoint": int(source_endpoint),
                        "attempt": attempt,
                        "changed_points": int(point_mask.sum()),
                        "changed_channels": int(channel_mask.sum()),
                    }
                )
                return SyntheticSample(
                    window=output.astype(np.float32),
                    point_mask=point_mask,
                    channel_mask=channel_mask,
                    metadata=metadata,
                )
        raise RuntimeError(
            f"Failed to generate a non-empty endpoint anomaly after "
            f"{self.config.max_generation_attempts} attempts"
        )
