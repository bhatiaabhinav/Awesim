import numpy as np
import torch.nn as nn

import bindings as A


def direction_onehot(direction) -> np.ndarray:
    onehot = np.zeros(4, dtype=np.float32)
    if direction == A.DIRECTION_NORTH:
        onehot[0] = 1.0
    elif direction == A.DIRECTION_EAST:
        onehot[1] = 1.0
    elif direction == A.DIRECTION_SOUTH:
        onehot[2] = 1.0
    elif direction == A.DIRECTION_WEST:
        onehot[3] = 1.0
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return onehot


def indicator_onehot(indicator) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    if indicator == A.INDICATOR_LEFT:
        onehot[0] = 1.0
    elif indicator == A.INDICATOR_NONE:
        onehot[1] = 1.0
    elif indicator == A.INDICATOR_RIGHT:
        onehot[2] = 1.0
    else:
        raise ValueError(f"Unknown indicator: {indicator}")
    return onehot


def make_mlp_with_layernorm(input_dim, hidden_units, output_dim):
    layers = []
    prev_dim = input_dim

    for units in hidden_units:
        layers.extend([
            nn.Linear(prev_dim, units),
            nn.LayerNorm(units),
            nn.ReLU()
        ])
        prev_dim = units

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


def make_mlp(input_dim, hidden_units, output_dim, layernorm=False):
    if layernorm:
        return make_mlp_with_layernorm(input_dim, hidden_units, output_dim)
    layers = []
    prev_dim = input_dim

    for units in hidden_units:
        layers.extend([
            nn.Linear(prev_dim, units),
            nn.ReLU()
        ])
        prev_dim = units

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)
    

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=k, stride=s,
            padding=p, dilation=d,
            bias=False
        )
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class VisionEncoder384(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (3, 384, 384)
            ConvBlock(3, 64, k=7, s=2, p=3),   # -> (64, 192, 192)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> (64, 96, 96)

            # Stage 1 (no downsample)
            ConvBlock(64,  64),                  # -> (64, 96, 96)
            ConvBlock(64,  64),                  # -> (64, 96, 96)

            # Stage 2
            ConvBlock(64, 128, s=2),             # -> (128, 48, 48)
            ConvBlock(128,128),                  # -> (128, 48, 48)
            ConvBlock(128,128),                  # -> (128, 48, 48)

            # Stage 3
            ConvBlock(128,256, s=2),             # -> (256, 24, 24)
            ConvBlock(256,256),                  # -> (256, 24, 24)
            ConvBlock(256,256),                  # -> (256, 24, 24)

            # Stage 4
            ConvBlock(256,512, s=2),             # -> (512, 12, 12)
            ConvBlock(512,512),                  # -> (512, 12, 12)
            ConvBlock(512,512),                  # -> (512, 12, 12)

            # Stage 5
            ConvBlock(512, output_dim, s=2),             # -> (output_dim, 6, 6)
            ConvBlock(output_dim, output_dim),           # -> (output_dim, 6, 6)
        )

        # From (output_dim, 6, 6) -> (output_dim, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)           # (B, output_dim, 6, 6)
        x = self.global_pool(x)        # (B, output_dim, 1, 1)
        return x


class LiteVisionEncoder384(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 3, 384, 384)

            # Conv1: 3 -> 32, 8x8, stride 4  -> (B, 32, 96, 96)
            ConvBlock(3, 32, k=8, s=4, p=2, d=1),

            # Conv2: 32 -> 64, 4x4, stride 2 -> (B, 64, 48, 48)
            ConvBlock(32, 64, k=4, s=2, p=1, d=1),

            # Conv3: 64 -> 128, 3x3, stride 2 -> (B, 128, 24, 24)
            ConvBlock(64, 128, k=3, s=2, p=1, d=1),

            # Conv4: 128 -> 256, 3x3, dilation 2 -> (B, 256, 24, 24)
            ConvBlock(128, 256, k=3, s=1, p=2, d=2),

            # Conv5: 256 -> 256, 3x3, dilation 4 -> (B, 256, 24, 24)
            ConvBlock(256, 256, k=3, s=1, p=4, d=4),

            # Conv6: 256 -> output_dim, 3x3, dilation 8 -> (B, output_dim, 24, 24)
            ConvBlock(256, output_dim, k=3, s=1, p=8, d=8),
        )

        # Global average pooling to (B, output_dim, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)          # (B, output_dim, 24, 24)
        x = self.global_pool(x)       # (B, output_dim, 1, 1)
        return x