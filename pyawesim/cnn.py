import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, hidden_dim: int, c_base: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, c_base, kernel_size=5,
                      stride=2, padding=2),
            nn.ReLU(inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(c_base, c_base * 2, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(c_base * 2, c_base * 4, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(c_base * 4, c_base * 4, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 8x8 -> 6x6
            nn.Conv2d(c_base * 4, c_base * 4, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(c_base * 4 * 6 * 6, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MinimalRLConv128(nn.Module):
    def __init__(self, hidden_dim: int, c_base=32):
        super().__init__()
        c2, c3, c4 = 2 * c_base, 2 * c_base, 4 * c_base  # 32,64,64,128 if c1=32

        self.net = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, c_base, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            # 64 -> 32
            nn.Conv2d(c_base, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32 -> 16
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 16 -> 8
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 8 -> 4
            nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Flatten(),                  # (B, c4 * 4 * 4)
            nn.Linear(c4 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def make_very_simple_downsamplingconv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )


class SimpleCNNWith3x3DownsamplingPairs(nn.Module):
    def __init__(self, hidden_dim: int, c_base: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            # 128x128 -> 64x64
            make_very_simple_downsamplingconv(3, c_base),

            # 64x64 -> 32x32
            make_very_simple_downsamplingconv(c_base, c_base * 2),

            # 32x32 -> 16x16
            make_very_simple_downsamplingconv(c_base * 2, c_base * 4),

            # 16x16 -> 8x8
            make_very_simple_downsamplingconv(c_base * 4, c_base * 4),

            # 8x8 -> 4x4
            make_very_simple_downsamplingconv(c_base * 4, c_base * 4),

            nn.Flatten(),
            nn.Linear(c_base * 4 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
