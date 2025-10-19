"""
(已修正) MobileNet-1D for ECG identification.
- 增加了 extract_features 方法，用于导出 embedding。
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNet1D(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        inverted_residual_setting: List[Tuple[int, int, int, int]] = [
            (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
        ]
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        features: List[nn.Module] = [ConvBNAct(1, input_channel, kernel_size=3, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNAct(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取分类器前的全局特征向量 (embeddings)，形状 [B, D]
        """
        x = self.features(x)        # [B, C, L']
        x = self.classifier[0](x)   # AdaptiveAvgPool1d(1) -> [B, C, 1]
        x = self.classifier[1](x)   # Flatten -> [B, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        修改后的 forward：先取 embedding，再走 Dropout + 全连接分类头
        """
        x = self.extract_features(x)  # [B, D]
        x = self.classifier[2](x)     # Dropout
        x = self.classifier[3](x)     # Linear -> logits
        return x