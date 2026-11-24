
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x) * 0.2


class BrandRestoreSR(nn.Module):
    def __init__(self, scale=4, channels=64, num_blocks=8):
        super().__init__()

        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        features = self.conv1(x)

        residual = self.residual_blocks(features)
        residual = self.conv2(residual)

        features = features + residual

        features = self.upsample(features)

        return torch.sigmoid(self.output(features))
