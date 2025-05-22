from torchvision.utils import save_image
from tqdm import tqdm

import torch
import torch.nn as nn
from dataset_loading import HZDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
import torch.optim as optim
import config


def train_fn(
    discriminator_monnet,
    discriminator_photo,
    generator_monnet,
    generator_photo,
    dataloader,
    opt_disc,
    opt_gen,
    L1,
    mse,
    g_scaler,
    d_scaler,
    epoch_idx,
):

    loop = tqdm(dataloader, leave=True)
    for idx, (photo, monnet) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        monnet = monnet.to(config.DEVICE)

        # train discriminators
        with torch.amp.autocast("cuda"):
            fake_monnet = generator_monnet(photo)
            disc_monnet_real = discriminator_monnet(monnet)
            disc_monnel_fake = discriminator_monnet(fake_monnet.detach())
            disc_monnet_real_loss = mse(
                disc_monnet_real, torch.ones_like(disc_monnet_real)
            )
            disc_monnet_fake_loss = mse(
                disc_monnel_fake, torch.zeros_like(disc_monnel_fake)
            )
            disc_monnet_loss = (disc_monnet_real_loss + disc_monnet_fake_loss) / 2

            fake_photo = generator_photo(monnet)
            disc_photo_real = discriminator_photo(photo)
            disc_photo_fake = discriminator_photo(fake_photo.detach())
            disc_photo_real_loss = mse(
                disc_photo_real, torch.ones_like(disc_photo_real)
            )
            disc_photo_fake_loss = mse(
                disc_photo_fake, torch.zeros_like(disc_photo_fake)
            )
            disc_photo_loss = (disc_photo_real_loss + disc_photo_fake_loss) / 2

            disc_loss = (disc_monnet_loss + disc_photo_loss) / 2
        print(f"Discriminator loss: {disc_loss.item()}")
        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generators
        with torch.amp.autocast("cuda"):
            # adversarial loss
            disc_monnet_fake = discriminator_monnet(fake_monnet)
            disc_photo_fake = discriminator_photo(fake_photo)
            gen_monnet_loss = mse(disc_monnet_fake, torch.ones_like(disc_monnet_fake))
            gen_photo_loss = mse(disc_photo_fake, torch.ones_like(disc_photo_fake))

            # cycle loss
            cycle_monnet = generator_monnet(fake_photo)
            cycle_photo = generator_photo(fake_monnet)
            cycle_monnet_loss = L1(monnet, cycle_monnet)
            cycle_photo_loss = L1(photo, cycle_photo)

            # identity loss
            identity_monnet = generator_monnet(monnet)
            identity_photo = generator_photo(photo)
            identity_monnet_loss = L1(monnet, identity_monnet)
            identity_photo_loss = L1(photo, identity_photo)

            gen_loss = (
                gen_photo_loss
                + gen_monnet_loss
                + cycle_monnet_loss * config.LAMBDA_CYCLE
                + cycle_photo_loss * config.LAMBDA_CYCLE
                + identity_monnet_loss * config.LAMBDA_IDENTITY
                + identity_photo_loss * config.LAMBDA_IDENTITY
            )
        print(f"Generator loss: {gen_loss.item()}")
        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

    config.save_image(
        fake_photo,
        filename=f"{config.IMAGES_ROOT}/fake_photo_{epoch_idx}.png",
    )
    config.save_image(
        fake_monnet,
        filename=f"{config.IMAGES_ROOT}/fake_monnet_{epoch_idx}.png",
    )


def main():
    discriminator_monnet = Discriminator(in_channels=3).to(config.DEVICE)
    discriminator_photo = Discriminator(in_channels=3).to(config.DEVICE)
    generator_monnet = Generator(in_channels=3, num_residuals=9).to(config.DEVICE)
    generator_photo = Generator(in_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(discriminator_monnet.parameters())
        + list(discriminator_photo.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(generator_monnet.parameters()) + list(generator_photo.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = HZDataset(
        root_H=f"{config.DATA_ROOT}/trainA",
        root_Z=f"{config.DATA_ROOT}/trainB",
        transform=config.transform,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # to support float 16 training
    g_scaler = torch.amp.GradScaler(device=config.DEVICE)
    d_scaler = torch.amp.GradScaler(device=config.DEVICE)

    for i in tqdm(range(config.NUM_EPOCHS)):
        train_fn(
            discriminator_monnet,
            discriminator_photo,
            generator_monnet,
            generator_photo,
            dataloader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            g_scaler,
            d_scaler,
            i,
        )


if __name__ == "__main__":
    main()
