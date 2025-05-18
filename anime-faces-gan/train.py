from data import AnimeDataset, transform
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Generator, Discriminator
import wandb
from tqdm import tqdm
import torch

wandb.init(
    project="GANS",
    name="AnimeGAN",
)

dataset = AnimeDataset(root_dir="./data.csv", transform=transform)
trainloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=0)


G = Generator(latent_size=128)
D = Discriminator()

lr = 0.0002
opt_d = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_g = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
adversarial_loss = torch.nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
G.to(device)
D.to(device)


def train_gan(num_epochs=10):
    for epoch in range(num_epochs):
        d_loss_global = 0
        g_loss_global = 0
        for i, real_images in enumerate(tqdm(trainloader)):
            # Train Discriminator
            real_images = real_images.to(device)

            real_disc = D(real_images)
            fake_images = G(torch.randn(real_images.size(0), 128, 1, 1).to(device))
            fake_disc = D(fake_images)

            loss_real = adversarial_loss(real_disc, torch.ones_like(real_disc))
            loss_fake = adversarial_loss(fake_disc, torch.zeros_like(fake_disc))
            d_loss = (loss_real + loss_fake) / 2

            D.zero_grad()
            d_loss.backward(retain_graph=True)
            opt_d.step()

            # Train Generator
            fake_disc_g = D(fake_images)
            g_loss = adversarial_loss(fake_disc_g, torch.ones_like(fake_disc))
            G.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_loss_global += d_loss.item()
            g_loss_global += g_loss.item()

            if i % 10 == 0:
                # print(
                #     f"Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}"
                # )
                wandb.log({"D Loss": d_loss.item(), "G Loss": g_loss.item()})

        d_loss_global = d_loss_global / len(trainloader)
        g_loss_global = g_loss_global / len(trainloader)

        print(
            f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}"
        )


if __name__ == "__main__":
    train_gan(num_epochs=10)

    # Save the model
    torch.save(G.state_dict(), "generator.pth")
    torch.save(D.state_dict(), "discriminator.pth")

    wandb.finish()
