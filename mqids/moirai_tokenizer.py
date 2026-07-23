"""Frozen MOIRAI encoder that returns one historical token per variable."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

NATIVE_PATCH_SIZES = (8, 16, 32, 64, 128)


class FrozenMoiraiTokenizer(nn.Module):
    def __init__(
        self,
        forecast: nn.Module,
        window_length: int,
        patch_size: int,
        target_dim: int,
        encoder_layer: int | None = None,
    ) -> None:
        super().__init__()
        if window_length != patch_size or patch_size not in NATIVE_PATCH_SIZES:
            raise ValueError("One-token mode requires equal native window and patch sizes")
        self.forecast = forecast
        self.window_length = int(window_length)
        self.patch_size = int(patch_size)
        self.target_dim = int(target_dim)
        num_layers = int(self.forecast.module.num_layers)
        if encoder_layer is not None and not 1 <= encoder_layer <= num_layers:
            raise ValueError(
                f"encoder_layer uses 1-based indexing and must be in [1, {num_layers}]"
            )
        self.encoder_layer = encoder_layer
        for parameter in self.forecast.parameters():
            parameter.requires_grad_(False)
        self.forecast.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        window_length: int,
        patch_size: int,
        target_dim: int,
        encoder_layer: int | None = None,
        device: torch.device | str,
    ) -> "FrozenMoiraiTokenizer":
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        module = MoiraiModule.from_pretrained(str(model_path)).to(device)
        forecast = MoiraiForecast(
            module=module,
            prediction_length=patch_size,
            context_length=window_length,
            patch_size=patch_size,
            num_samples=1,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(device)
        return cls(forecast, window_length, patch_size, target_dim, encoder_layer)

    def train(self, mode: bool = True) -> "FrozenMoiraiTokenizer":
        super().train(mode)
        self.forecast.eval()
        return self

    def _encode_selected_layer(
        self,
        representations: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        time_id: torch.Tensor,
        variate_id: torch.Tensor,
    ) -> torch.Tensor:
        encoder = self.forecast.module.encoder
        if self.encoder_layer is None:
            return encoder(
                representations,
                attention_mask,
                time_id=time_id,
                var_id=variate_id,
            )
        selected_index = self.encoder_layer - 1
        if encoder.use_moe:
            for index, layer in enumerate(encoder.layers):
                representations = layer(
                    representations,
                    attention_mask,
                    var_id=variate_id,
                    time_id=time_id,
                    centroid=encoder.centroid[index],
                )
                if index == selected_index:
                    break
        else:
            for index, layer in enumerate(encoder.layers):
                representations = layer(
                    representations,
                    attention_mask,
                    var_id=variate_id,
                    time_id=time_id,
                )
                if index == selected_index:
                    break
        return encoder.norm(representations)

    @torch.no_grad()
    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError(f"Expected [batch, time, variables], got {tuple(windows.shape)}")
        if windows.shape[1:] != (self.window_length, self.target_dim):
            raise ValueError(
                f"Expected trailing shape {(self.window_length, self.target_dim)}, "
                f"got {tuple(windows.shape[1:])}"
            )
        observed = torch.isfinite(windows)
        windows = torch.nan_to_num(windows)
        is_pad = torch.zeros(windows.shape[:2], dtype=torch.bool, device=windows.device)
        target, observed_mask, sample_id, time_id, variate_id, prediction_mask = self.forecast._convert(
            self.patch_size,
            past_target=windows,
            past_observed_target=observed,
            past_is_pad=is_pad,
        )
        module = self.forecast.module
        loc, scale = module.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        representations = module.in_proj(
            scaled_target,
            torch.full_like(time_id, self.patch_size, dtype=torch.long),
        )
        from uni2ts.common.torch_util import mask_fill, packed_attention_mask

        representations = mask_fill(representations, prediction_mask, module.mask_encoding.weight)
        representations = self._encode_selected_layer(
            representations,
            packed_attention_mask(sample_id),
            time_id=time_id,
            variate_id=variate_id,
        )

        context_tokens: list[torch.Tensor] = []
        for batch_index in range(windows.shape[0]):
            context_indices = torch.where(~prediction_mask[batch_index])[0]
            context_var_ids = variate_id[batch_index, context_indices]
            order = torch.argsort(context_var_ids)
            selected = representations[batch_index, context_indices[order]]
            selected_ids = context_var_ids[order]
            expected_ids = torch.arange(self.target_dim, device=selected_ids.device)
            if selected.shape[0] != self.target_dim or not torch.equal(selected_ids, expected_ids):
                raise RuntimeError(
                    "MOIRAI conversion did not produce exactly one ordered context token per variable; "
                    f"got {selected.shape[0]} tokens for {self.target_dim} variables"
                )
            context_tokens.append(selected)
        return torch.stack(context_tokens, dim=0).float()
