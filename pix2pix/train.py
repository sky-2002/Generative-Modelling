# Training

from tqdm import tqdm
import torch
from torch import nn
from generator_model import Generator
from discriminator_model import Discriminator
from data import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

images_inter = []


def train(disc, gen, dataloader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):

    loop = tqdm(dataloader, leave=True)

    for id, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda"):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.amp.autocast("cuda"):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            l1_loss = l1(y_fake, y) * 100
            G_loss = G_fake_loss + l1_loss
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

    images_inter.append(y_fake)
    print(f"Discriminator loss: {D_loss.item()}, Generator loss: {G_loss.item()}")


def main():
    disc = Discriminator(3).to(device)
    gen = Generator(3).to(device)

    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))

    BCE_LOSS = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    g_scaler = torch.amp.GradScaler(device=device)
    d_scaler = torch.amp.GradScaler(device=device)

    for epoch in range(5):
        train(
            disc,
            gen,
            dataloader,
            opt_disc,
            opt_gen,
            L1_LOSS,
            BCE_LOSS,
            g_scaler,
            d_scaler,
        )
    return disc, gen


disc, gen = main()
