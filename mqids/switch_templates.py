"""Stable endpoint-state text for discrete WADI variables."""

from __future__ import annotations

import numpy as np
import torch

from .data import DiscreteStateVocabulary


def _lookup_state(
    current_value: float,
    values: tuple[float, ...],
    names: tuple[str, ...],
    *,
    scaler_mean: float,
    scaler_scale: float,
) -> str:
    candidates = np.asarray(values, dtype=np.float64)
    matches = np.flatnonzero(np.isclose(candidates, current_value, rtol=0.0, atol=1e-6))
    if matches.size:
        return names[int(matches[0])]
    raw_value = current_value * scaler_scale + scaler_mean
    return f"未知状态（原始值={raw_value:g}）"


def build_discrete_text(
    discrete_windows: torch.Tensor,
    vocabulary: DiscreteStateVocabulary,
) -> tuple[str, ...]:
    """Return endpoint-only state text in vocabulary order for one sample.

    The earlier implementation remapped states independently inside every
    window. This function deliberately uses only the fixed vocabulary inferred
    from normal training data and reads only the last value in each window.
    """
    windows = discrete_windows.detach().cpu().numpy()
    if windows.ndim == 1:
        windows = windows.reshape(1, -1)
    if windows.shape[0] != len(vocabulary.variables):
        raise ValueError("discrete_windows and vocabulary must have matching lengths")
    if windows.shape[1] == 0:
        raise ValueError("A discrete window cannot be empty")

    return tuple(
        _lookup_state(
            float(windows[i, -1]),
            spec.values,
            spec.state_names,
            scaler_mean=spec.scaler_mean,
            scaler_scale=spec.scaler_scale,
        )
        for i, spec in enumerate(vocabulary.variables)
    )
