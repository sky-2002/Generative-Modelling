# Discriminator model
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=4,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                stride=2,
                kernel_size=4,
                padding_mode="reflect",
                padding=1,
            ),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))

    model = Discriminator()
    out = model(x, y)
    print(out.shape)
