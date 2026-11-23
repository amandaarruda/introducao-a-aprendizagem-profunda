"""MMDetection backbone adapter for OpenCLIP's ModifiedResNet (RN101)."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class OpenCLIPModifiedResNet(nn.Module):
    """Expose selected spatial stages of OpenAI-CLIP's RN101 visual encoder."""

    def __init__(
        self,
        checkpoint: str,
        frozen_stages: int = -1,
        norm_eval: bool = False,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        import open_clip

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"CLIP_RN101_CHECKPOINT does not exist: {checkpoint}")
        model = open_clip.create_model("RN101", pretrained=None)
        state = torch.load(checkpoint, map_location="cpu")
        # OpenAI's published RN101 file is TorchScript; other exports may be state dicts.
        if not isinstance(state, Mapping):
            state = state.state_dict()
        state = state.get("state_dict", state)
        state = {key: value for key, value in state.items() if key in model.state_dict()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected CLIP checkpoint keys: {unexpected[:5]}")
        self.visual = model.visual
        self.out_channels = [256, 512, 1024, 2048]
        self.frozen_stages, self.norm_eval = frozen_stages, norm_eval
        if not out_indices or any(index not in range(4) for index in out_indices):
            raise ValueError("out_indices must select one or more CLIP ResNet stages")
        self.out_indices = tuple(out_indices)
        if missing:
            raise RuntimeError(f"incomplete CLIP RN101 load; missing keys: {missing[:5]}")

    def forward(self, x):
        v = self.visual
        x = v.act1(v.bn1(v.conv1(x)))
        x = v.act2(v.bn2(v.conv2(x)))
        x = v.act3(v.bn3(v.conv3(x)))
        x = v.avgpool(x)
        c2 = v.layer1(x)
        c3 = v.layer2(c2)
        c4 = v.layer3(c3)
        c5 = v.layer4(c4)
        stages = (c2, c3, c4, c5)
        return tuple(stages[index] for index in self.out_indices)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self
