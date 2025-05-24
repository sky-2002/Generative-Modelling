import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, down=True, use_act=True, **kwargs
    ):
        super().__init__()

        self.conv = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    padding_mode="reflect",
                    **kwargs,
                )
                if down
                else nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    **kwargs,
                )
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(0.2) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_act=False,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    kernel_size=3,
                    stride=2,
                    down=False,
                    padding=1,
                    # output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features,
                    kernel_size=3,
                    stride=2,
                    down=False,
                    padding=1,
                    # output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features,
            in_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.residual_blocks(x)
        for block in self.up_blocks:
            x = block(x)
        return torch.tanh(self.last(x))


if __name__ == "__main__":
    # Test the generator
    gen = Generator()
    print(f"Number of parameters: {sum(p.numel() for p in gen.parameters())}")
    x = torch.randn(1, 3, 256, 256)
    out = gen(x)
    print(out.shape)  # Should be (1, 3, 256, 256)
