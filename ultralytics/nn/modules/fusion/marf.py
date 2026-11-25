"""Misalignment-aware reliability fusion for paired RGB and infrared features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["MARF"]


class MARF(nn.Module):
    """Align an auxiliary modality and fuse it with a reliability-aware gate.

    The first input is the geometrical reference and the second input is warped
    towards it. The offset head starts at zero, while the reliability gate starts
    with a reference-favoring prior for stable optimization.

    Args:
        inc: Input channels in ``[reference, auxiliary]`` order.
        ouc: Output channels.
        max_offset: Maximum displacement in feature-map pixels.
        reduction: Hidden-channel reduction ratio.
        reference_prior: Initial logit magnitude favoring the first input.
        force_fp32_warp: Run grid sampling in FP32 under mixed precision.
    """

    def __init__(
        self,
        inc: list[int],
        ouc: int,
        max_offset: float = 2.0,
        reduction: int = 4,
        reference_prior: float = 0.5,
        force_fp32_warp: bool = True,
    ) -> None:
        super().__init__()
        if len(inc) != 2:
            raise ValueError(f"MARF expects exactly two input channels, got {len(inc)}")
        if ouc <= 0:
            raise ValueError(f"MARF output channels must be positive, got {ouc}")
        if max_offset < 0:
            raise ValueError(f"MARF max_offset must be non-negative, got {max_offset}")
        if reduction <= 0:
            raise ValueError(f"MARF reduction must be positive, got {reduction}")

        self.max_offset = float(max_offset)
        self.force_fp32_warp = bool(force_fp32_warp)
        hidden = max(ouc // int(reduction), 16)

        self.reference_align = Conv(inc[0], ouc, 1) if inc[0] != ouc else nn.Identity()
        self.auxiliary_align = Conv(inc[1], ouc, 1) if inc[1] != ouc else nn.Identity()

        # Predict auxiliary-to-reference displacement from cross-modal disagreement.
        offset_channels = ouc * 3
        self.offset_encoder = nn.Sequential(
            Conv(offset_channels, offset_channels, 3, g=offset_channels),
            Conv(offset_channels, hidden, 1),
        )
        self.offset_head = nn.Conv2d(hidden, 2, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        # Estimate per-pixel, per-channel reliability after spatial alignment.
        interaction_channels = ouc * 4
        self.gate_encoder = nn.Sequential(
            Conv(interaction_channels, interaction_channels, 3, g=interaction_channels),
            Conv(interaction_channels, hidden, 1),
        )
        self.gate_head = nn.Conv2d(hidden, ouc * 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate_head.weight)
        with torch.no_grad():
            self.gate_head.bias[:ouc].fill_(float(reference_prior))
            self.gate_head.bias[ouc:].fill_(-float(reference_prior))

        # Retain complementary information not represented by reliability weights.
        self.complementary = nn.Sequential(
            Conv(ouc * 2, ouc, 1),
            Conv(ouc, ouc, 3, g=ouc),
        )
        self.complementary_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _sampling_grid(offset: torch.Tensor) -> torch.Tensor:
        """Convert pixel displacement to a normalized grid for align_corners=False."""
        batch, _, height, width = offset.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=offset.device, dtype=offset.dtype),
            torch.arange(width, device=offset.device, dtype=offset.dtype),
            indexing="ij",
        )
        base_x = (2.0 * (x + 0.5) / width) - 1.0
        base_y = (2.0 * (y + 0.5) / height) - 1.0
        base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        offset_x = 2.0 * offset[:, 0] / width
        offset_y = 2.0 * offset[:, 1] / height
        return base_grid + torch.stack((offset_x, offset_y), dim=-1)

    def _warp(self, auxiliary: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        grid = self._sampling_grid(offset)
        output_dtype = auxiliary.dtype
        use_fp32 = self.force_fp32_warp and output_dtype in (torch.float16, torch.bfloat16)
        if use_fp32:
            auxiliary, grid = auxiliary.float(), grid.float()
        aligned = F.grid_sample(
            auxiliary,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return aligned.to(output_dtype) if use_fp32 else aligned

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("MARF expects [reference, auxiliary] feature maps")
        reference, auxiliary = inputs
        if reference.shape[-2:] != auxiliary.shape[-2:]:
            raise ValueError(
                "MARF inputs must share a spatial shape, "
                f"got {reference.shape[-2:]} and {auxiliary.shape[-2:]}"
            )

        reference = self.reference_align(reference)
        auxiliary = self.auxiliary_align(auxiliary)

        initial_difference = torch.abs(reference - auxiliary)
        offset_features = torch.cat((reference, auxiliary, initial_difference), dim=1)
        offset = torch.tanh(self.offset_head(self.offset_encoder(offset_features))) * self.max_offset
        aligned_auxiliary = self._warp(auxiliary, offset)

        difference = torch.abs(reference - aligned_auxiliary)
        consistency = reference * aligned_auxiliary
        interaction = torch.cat((reference, aligned_auxiliary, difference, consistency), dim=1)
        gate_logits = self.gate_head(self.gate_encoder(interaction))
        batch, _, height, width = gate_logits.shape
        gates = torch.softmax(gate_logits.view(batch, 2, -1, height, width), dim=1)
        reference_gate, auxiliary_gate = gates.unbind(dim=1)

        reliable = reference_gate * reference + auxiliary_gate * aligned_auxiliary
        complementary = self.complementary(torch.cat((difference, consistency), dim=1))
        return reliable + torch.tanh(self.complementary_scale) * complementary
