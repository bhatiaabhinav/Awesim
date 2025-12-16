import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, hidden_dim: int, c_in: int = 3, c_base: int = 32, groupnorm=False, num_groups=32):
        super().__init__()

        if c_in % 3 != 0:
            raise ValueError(
                "c_in must be divisible by 3 (stacked RGB frames)")

        self.num_frames = c_in // 3
        self.c_in = c_in  # stored for reference / checks

        # Shared stem: first two conv layers applied separately to each frame (weight sharing)
        self.stem = nn.Sequential(
            # 128x128x3 -> 64x64xc_base
            nn.Conv2d(3, c_base, kernel_size=5, stride=2, padding=2),
            nn.Identity() if not groupnorm else nn.GroupNorm(min(num_groups, c_base), c_base),
            nn.ReLU(inplace=True),

            # 64x64xc_base -> 32x32x(c_base*2)
            nn.Conv2d(c_base, c_base * 2, kernel_size=4, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 2), c_base * 2),
            nn.ReLU(inplace=True),
        )

        # Remaining network operating on temporally stacked features
        self.net = nn.Sequential(
            # 32x32x(c_base*2 * num_frames) -> 16x16x(c_base*4)
            nn.Conv2d(c_base * 2 * self.num_frames, c_base * 4, kernel_size=4, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 4), c_base * 4),
            nn.ReLU(inplace=True),

            # 16x16x(c_base*4) -> 8x8x(c_base*4)
            nn.Conv2d(c_base * 4, c_base * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 4), c_base * 4),
            nn.ReLU(inplace=True),

            # 8x8x(c_base*4) -> 6x6x(c_base*4)
            nn.Conv2d(c_base * 4, c_base * 4,
                      kernel_size=3, stride=1, padding=0),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 4), c_base * 4),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(c_base * 4 * 6 * 6, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, c_in, H, W) where c_in = 3 * num_frames
        B, C, H, W = x.shape

        # Reshape to process each frame independently through the shared stem
        x = x.reshape(B * self.num_frames, 3, H, W)
        x = self.stem(x)  # (B * num_frames, c_base * 2, 32, 32)

        # Stack features along channel dimension for temporal mixing
        # (B, c_base * 2 * num_frames, 32, 32)
        x = x.view(B, -1, x.shape[2], x.shape[3])

        return self.net(x)


class MinimalCNN(nn.Module):
    def __init__(self, hidden_dim: int, c_in=3, c_base=32, groupnorm=False, num_groups=32):
        super().__init__()

        if c_in % 3 != 0:
            raise ValueError(
                "c_in must be divisible by 3 (stacked RGB frames)")

        self.num_frames = c_in // 3
        self.c_in = c_in  # stored for reference / checks

        # Shared stem: first two conv layers applied separately to each frame (weight sharing)
        self.stem = nn.Sequential(
            # 128x128x3 -> 64x64xc_base
            nn.Conv2d(3, c_base, kernel_size=5, stride=2, padding=2),
            nn.Identity() if not groupnorm else nn.GroupNorm(min(num_groups, c_base), c_base),
            nn.ReLU(inplace=True),

            # 64x64xc_base -> 32x32x(c_base*2)
            nn.Conv2d(c_base, c_base * 2, kernel_size=3, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 2), c_base * 2),
            nn.ReLU(inplace=True),
        )

        # Remaining network operating on temporally stacked features
        self.net = nn.Sequential(
            # 32x32x(c_base*2 * num_frames) -> 16x16x(c_base*4)
            nn.Conv2d(c_base * 2 * self.num_frames, c_base * 4, kernel_size=3, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 4), c_base * 4),
            nn.ReLU(inplace=True),

            # 16x16x(c_base*4) -> 8x8x(c_base*4)
            nn.Conv2d(c_base * 4, c_base * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 4), c_base * 4),
            nn.ReLU(inplace=True),

            # 8x8x(c_base*4) -> 4x4x(c_base*4)
            nn.Conv2d(c_base * 4, c_base * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(
                min(num_groups, c_base * 4), c_base * 4),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(c_base * 4 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, c_in, H, W) where c_in = 3 * num_frames
        B, C, H, W = x.shape
        x = x.reshape(B * self.num_frames, 3, H, W)
        x = self.stem(x)  # (B * num_frames, c_base * 2, 32, 32)

        # Stack features along channel dimension for temporal mixing
        x = x.view(B, -1, x.shape[2], x.shape[3])

        return self.net(x)


class DownsampleConvBlock(nn.Module):
    def __init__(self, c_in, c_out, groupnorm=False, num_groups=32, residual=False):
        super().__init__()
        self.residual = residual
        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(min(num_groups, c_in), c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
            nn.Identity() if not groupnorm else nn.GroupNorm(min(num_groups, c_out), c_out),
        )
        if self.residual:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0),
                nn.Identity() if not groupnorm else nn.GroupNorm(min(num_groups, c_out), c_out),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x)
        if self.residual:
            out += self.shortcut(x)
        return self.relu(out)


class DeepCNN(nn.Module):
    def __init__(self, hidden_dim: int, c_in=3, c_base: int = 32, groupnorm=False, num_groups=32, residual=False):
        super().__init__()

        if c_in % 3 != 0:
            raise ValueError(
                "c_in must be divisible by 3 (stacked RGB frames)")

        self.num_frames = c_in // 3
        self.c_in = c_in  # stored for reference / checks

        # Shared stem: first two downsampling blocks applied separately to each frame (weight sharing)
        self.stem = nn.Sequential(
            # 128x128x3 -> 64x64xc_base
            DownsampleConvBlock(3, c_base, groupnorm=groupnorm,
                                num_groups=num_groups, residual=residual),

            # 64x64xc_base -> 32x32x(c_base*2)
            DownsampleConvBlock(c_base, c_base * 2, groupnorm=groupnorm,
                                num_groups=num_groups, residual=residual),
        )

        # Remaining network operating on temporally stacked features
        self.net = nn.Sequential(
            # 32x32x(c_base*2 * num_frames) -> 16x16x(c_base*4)
            DownsampleConvBlock(c_base * 2 * self.num_frames, c_base * 4,
                                groupnorm=groupnorm, num_groups=num_groups, residual=residual),

            # 16x16x(c_base*4) -> 8x8x(c_base*4)
            DownsampleConvBlock(c_base * 4, c_base * 4, groupnorm=groupnorm,
                                num_groups=num_groups, residual=residual),

            # 8x8x(c_base*4) -> 4x4x(c_base*4)
            DownsampleConvBlock(c_base * 4, c_base * 4, groupnorm=groupnorm,
                                num_groups=num_groups, residual=residual),

            nn.Flatten(),
            nn.Linear(c_base * 4 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, c_in, H, W) where c_in = 3 * num_frames
        B, C, H, W = x.shape
        x = x.reshape(B * self.num_frames, 3, H, W)
        x = self.stem(x)  # (B * num_frames, c_base * 2, 32, 32)

        # Stack features along channel dimension for temporal mixing
        x = x.view(B, -1, x.shape[2], x.shape[3])

        return self.net(x)


def DeepResidualCNN(hidden_dim: int, c_in=3, c_base: int = 32, groupnorm=False, num_groups=32):
    return DeepCNN(hidden_dim, c_in, c_base, groupnorm, num_groups, residual=True)
