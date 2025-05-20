from model import VAE

from data import AnimeDataset, transform
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
from tqdm import tqdm
import torch

# wandb.init(
#     project="GANS",
#     name="Anime-VAE",
# )

dataset = AnimeDataset(root_dir="../anime-faces-gan/data.csv", transform=transform)
trainloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vae = VAE(latent_dim=128).to(device)

vae_optimizer = Adam(vae.parameters(), lr=0.0002, betas=(0.5, 0.999))


def train(epochs=5):
    for epoch in range(epochs):
        loss_global = 0
        for i, real_images in enumerate(tqdm(trainloader)):
            real_images = real_images.to(device)
            vae_optimizer.zero_grad()
            recon_images, mu, logvar = vae(real_images)
            loss = vae.loss_function(recon_images, real_images, mu, logvar)
            loss.backward()
            vae_optimizer.step()
            if i % 10 == 0:
                # print(
                #     f"Epoch [{epoch}/{epochs}], Step [{i}/{len(trainloader)}], Loss: {loss.item():.4f}"
                # )
                pass
                # wandb.log({"Loss": loss.item()})
            loss_global += loss.item()
        print(
            f"Epoch [{epoch}/{epochs}], Average Loss: {loss_global / len(trainloader):.4f}"
        )
        torch.save(
            vae.state_dict(),
            f"vae_epoch_{epoch}.pth",
        )


if __name__ == "__main__":
    train(epochs=5)
    torch.save(vae.state_dict(), "vae.pth")
    print("Model saved to vae.pth")
    # wandb.finish()
