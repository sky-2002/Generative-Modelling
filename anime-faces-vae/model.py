import torch.nn as nn
import torch
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = Sampling(mu, logvar)()
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.mse_loss(recon_x.view(-1), x.view(-1), reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


class Sampling(nn.Module):
    def __init__(self, mu, logvar):
        super(Sampling, self).__init__()
        self.mu = mu
        self.logvar = logvar

    def forward(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(self.mu)


class Encoder(nn.Module):

    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.conv1 = self.block(3, 64)  # 64 -> 32
        self.conv2 = self.block(64, 128)  # 32 -> 16
        self.conv3 = self.block(128, 256)  # 16 -> 8

        self.mu_linear = nn.Linear(256 * 8 * 8, latent_dim)
        self.logvar_linear = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)
        return mu, logvar

    def block(
        self,
        in_channels,
        out_channels,
    ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class Decoder(nn.Module):

    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.latent_to_hidden = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            self.deconv_block(512, 512),  # 4 -> 8
            self.deconv_block(512, 256),  # 8 -> 16
            self.deconv_block(256, 128),  # 16 -> 32
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.latent_to_hidden(z)
        z = z.view(z.size(0), 512, 4, 4)
        z = self.deconv(z)
        return z

    def deconv_block(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


if __name__ == "__main__":
    # Test the Encoder and Decoder
    encoder = Encoder(latent_dim=128)
    decoder = Decoder(latent_dim=128)

    # Create a random input tensor
    input_tensor = torch.randn(8, 3, 64, 64)  # Batch size of 8, 3 channels, 64x64 image

    # Forward pass through the encoder
    mu, logvar = encoder(input_tensor)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)

    # Sample from the latent space
    sampling = Sampling(mu, logvar)
    z = sampling()
    print("Sampled z shape:", z.shape)

    # # Forward pass through the decoder
    output_tensor = decoder(z)
    print("Output tensor shape:", output_tensor.shape)
