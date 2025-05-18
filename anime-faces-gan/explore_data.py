from PIL import Image
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def make_array():
    arr = []
    # Randomly select 64 images to visualize
    for i in range(64):
        random_image = random.choice(image_list)
        arr.append(np.asarray(Image.open(random_image).convert("RGB")))
    return np.array(arr)


if __name__ == "__main__":

    image_list = []
    rows = []
    for filename in glob.glob("../anime/images/*.jpg"):
        im = Image.open(filename)
        rows.append([filename])
        image_list.append(filename)
    print(f"Number of images: {len(image_list)}")

    df = pd.DataFrame(rows)
    df.to_csv("data.csv", index=False, header=None)

    array = make_array()
    result = gallery(array)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(result)
    plt.show()
