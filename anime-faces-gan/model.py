import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_size: int = 128):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_size,
                out_channels=512,
                kernel_size=4,
            ),  # output shape: (512, 4, 4)
            nn.BatchNorm2d(num_features=512),  # one gamma, one beta for each channel
            nn.ReLU(),
            # using weight -> Batch normalization -> activation has advantages
            # 1. Inputs to non-linear function should be zero mean, unit variance,
            # because it keeps them in a region where gradient does not saturate
            # 2. After training, gamma and beta can be absored into preceding weights and bias
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # output shape: (256, 8, 8)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # output shape: (128, 16, 16)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # output shape: (64, 32, 32)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # output shape: (3, 64, 64)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        return x  # (3, 64, 64)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=0
            ),
            nn.Sigmoid(),
        )

    def make_spectral(self):
        self.conv_block1 = nn.Sequential(
            self.add_block(3, 64, 4, 2, 1, spectral=True),
            self.add_block(64, 128, 4, 2, 1, spectral=True),
            self.add_block(128, 256, 4, 2, 1, spectral=True),
            self.add_block(256, 512, 4, 2, 1, spectral=True),
            self.add_block(512, 1, 4, 2, 0, spectral=True, last_layer=True),
        )

    def add_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        spectral=False,
        last_layer=False,
    ):
        if not last_layer:
            block = nn.Sequential(
                (
                    nn.utils.spectral_norm(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                        )
                    )
                    if spectral
                    else nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            block = nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ),
                nn.Sigmoid(),
            )
        return block

    def forward(self, x):
        x = self.conv_block1(x)
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import denorm
    from explore_data import gallery

    # Test the generator
    latent_size = 128
    generator = Generator(latent_size)
    noise = torch.randn(1, latent_size, 1, 1)

    fake = generator(noise)
    print(fake.shape)  # should be (1, 3, 64, 64)
    # Test the discriminator
    discriminator = Discriminator()
    d_fake = discriminator(fake)
    print(d_fake.shape, d_fake)  # should be (1, 1, 1, 1)

    fake_images = generator(torch.randn(64, latent_size, 1, 1)).detach()

    def show_images(images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        denormed_images = denorm(images)
        denormed_images = denormed_images.permute(0, 2, 3, 1).numpy()
        n = min(nmax, len(denormed_images))
        grid = gallery(denormed_images[:n], ncols=8)
        ax.imshow(grid)
        plt.axis("off")
        plt.show()

    show_images(fake_images)
