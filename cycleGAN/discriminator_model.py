import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),  # Why reflect?
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),  # ReLU for generator, LeakyReLU for discriminator
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.layers = []
        in_channels = features[0]

        for feature in features[1:]:
            self.layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        self.layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    # Test the Discriminator
    discriminator = Discriminator()
    print(f"Number of parameters: {sum(p.numel() for p in discriminator.parameters())}")
    x = torch.randn(1, 3, 256, 256)  # Example input
    output = discriminator(x)
    print(output.shape)  # Should be (1, 1, 30, 30) for a 256x256 input
