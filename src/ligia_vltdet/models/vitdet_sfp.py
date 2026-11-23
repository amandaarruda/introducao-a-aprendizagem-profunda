"""ViTDet-style Simple Feature Pyramid for a single CLIP RN101 feature map."""
from __future__ import annotations

import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class ViTDetSimpleFeaturePyramid(nn.Module):
    """Turn a final stride-32 encoder feature into four pyramid levels."""

    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 256,
        scale_factors: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in scale_factors:
            if scale == 4.0:
                mid1, mid2 = in_channels // 2, in_channels // 4
                stage = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, mid1, 2, stride=2),
                    nn.GroupNorm(norm_groups, mid1), nn.GELU(),
                    nn.ConvTranspose2d(mid1, mid2, 2, stride=2),
                    nn.GroupNorm(norm_groups, mid2), nn.GELU(),
                    nn.Conv2d(mid2, out_channels, 1),
                )
            elif scale == 2.0:
                stage = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                    nn.GroupNorm(norm_groups, out_channels), nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 1),
                )
            elif scale == 1.0:
                stage = nn.Conv2d(in_channels, out_channels, 1)
            elif scale == 0.5:
                stage = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels, out_channels, 1),
                )
            else:
                raise ValueError(f"unsupported ViTDet scale factor: {scale}")
            self.stages.append(stage)

    def forward(self, inputs):
        if len(inputs) != 1:
            raise ValueError("ViTDet SFP expects only the final encoder feature")
        feature = inputs[0]
        return tuple(stage(feature) for stage in self.stages)
