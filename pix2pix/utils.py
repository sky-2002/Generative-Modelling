import matplotlib.pyplot as plt


def plot_gen(gen, dataloader, rows=1):
    imgs_batch, masks_batch = next(iter(dataloader))
    print(imgs_batch.shape)

    plt.figure(figsize=(12, 24))
    for i, (img, mask) in enumerate(zip(imgs_batch[:rows], masks_batch[:rows])):
        generated = gen(img.unsqueeze(0)).squeeze(0).detach()
        img = (img.permute(1, 2, 0).numpy() + 1) / 2
        mask = (mask.permute(1, 2, 0).numpy() + 1) / 2

        generated = (generated.permute(1, 2, 0).numpy() + 1) / 2
        plt.subplot(rows, 3, 3 * i + 1)
        plt.title("input")
        plt.imshow(img)

        plt.subplot(rows, 3, 3 * i + 2)
        plt.title("Actual")
        plt.imshow(mask)

        plt.subplot(rows, 3, 3 * i + 3)
        plt.title("Generated")
        plt.imshow(generated)

        plt.show()


def explore_data(dataloader):
    imgs_batch, masks_batch = next(iter(dataloader))

    plt.figure(figsize=(12, 24))
    for i, (img, mask) in enumerate(zip(imgs_batch[:3], masks_batch[:3])):
        img = (img.permute(1, 2, 0).numpy() + 1) / 2
        mask = (mask.permute(1, 2, 0).numpy() + 1) / 2
        plt.subplot(3, 2, 2 * i + 1)
        plt.title("input")
        plt.imshow(img)
        plt.subplot(3, 2, 2 * i + 2)
        plt.title("output")
        plt.imshow(mask)

    plt.show()
