import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms
import glob
from PIL import Image

imgs_path = glob.glob("../input/anime-sketch-colorization-pair/data/train/*.png")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 512)),
        transforms.Normalize(0.5, 0.5),
    ]
)


class Anime_dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert("RGB")
        pil_img = transform(pil_img)
        w = pil_img.size(2) // 2
        return pil_img[:, :, w:], pil_img[:, :, :w]

    def __len__(self):
        return len(self.imgs_path)


dataset = Anime_dataset(imgs_path)

batchsize = 128
dataloader = data.DataLoader(
    dataset, batch_size=batchsize, shuffle=True, pin_memory=True
)
