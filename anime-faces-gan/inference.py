from model import Generator
import torch
import matplotlib.pyplot as plt
from data import denorm
from explore_data import gallery


G = Generator(latent_size=128)
# load from .pth file
G.load_state_dict(torch.load("generator.pth"))

noise = torch.randn(64, 128, 1, 1)


if __name__ == "__main__":

    images = G(noise)
    images = images.detach()

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

    show_images(images)
