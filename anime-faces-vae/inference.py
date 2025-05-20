from model import VAE
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from data import denorm


def gallery(array, ncols=8):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    result = (
        array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )
    return result


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


def plot_samples(all_samples, n_samples=5, batch_size=8, nrow=8, figsize=(15, 30)):
    """Plot samples from each model, with epoch annotations only on first row."""
    fig, axs = plt.subplots(n_samples, len(all_samples), figsize=figsize)

    # If we only have one row, ensure axs is still 2D
    if n_samples == 1:
        axs = np.expand_dims(axs, axis=0)

    for epoch_idx, epoch_samples in enumerate(all_samples):
        for sample_idx, samples in enumerate(epoch_samples):
            # Convert from tensor to numpy for plotting
            # Apply denormalization
            denormed_samples = denorm(samples)
            samples_grid = (
                make_grid(denormed_samples, nrow=nrow).permute(1, 2, 0).numpy()
            )

            # Plot on the corresponding axis
            ax = axs[sample_idx, epoch_idx]
            ax.imshow(samples_grid)

            # Only add epoch title to the first row
            if sample_idx == 0:
                ax.set_title(f"Epoch {epoch_idx}")

            ax.axis("off")

    # Remove whitespace between rows only
    plt.subplots_adjust(hspace=0)

    # Save and show
    plt.savefig("generated_samples_by_epoch.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Parameters
    latent_dim = 128
    batch_size = 8
    n_samples = 5
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Storage for samples from all epochs
    all_samples = []

    # For each epoch
    for epoch in range(num_epochs):
        # Load the model for this epoch
        vae = VAE(latent_dim=latent_dim)
        vae.load_state_dict(torch.load(f"vae_epoch_{epoch}.pth"))
        vae.eval()
        vae = vae.to(device)

        # Store samples for this epoch
        epoch_samples = []

        # Generate 5 batches of 8 images
        for i in range(n_samples):
            # Generate random noise
            noise = torch.randn(batch_size, 1, latent_dim, device=device)

            # Generate images
            with torch.no_grad():
                images = vae.decoder(noise)

            # Move to CPU
            images = images.detach().cpu()

            # Add to our collection
            epoch_samples.append(images)

        # Add all samples from this epoch to our collection
        all_samples.append(epoch_samples)

    # Plot all samples
    # print("Plotting samples from all epochs...")
    # plot_samples(all_samples, n_samples=n_samples, batch_size=batch_size)

    epoch = 4
    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(torch.load(f"vae_epoch_{epoch}.pth"))
    vae.eval()

    vae = vae.to(device)
    noise = torch.randn(64, 1, latent_dim, device=device)
    with torch.no_grad():
        images = vae.decoder(noise)
    images = images.detach().cpu()

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

    show_images(images, nmax=64)
    print("Done!")
