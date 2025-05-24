from torch.utils.data import Dataset, DataLoader

import os
from PIL import Image
import numpy as np


class HZDataset(Dataset):
    def __init__(self, root_H, root_Z, transform=None):
        self.root_H = root_H
        self.root_Z = root_Z
        self.transform = transform

        self.H_images = os.listdir(root_H)
        self.Z_images = os.listdir(root_Z)
        self.length_dataset = max(len(self.H_images), len(self.Z_images))
        self.H_len = len(self.H_images)
        self.Z_len = len(self.Z_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        Z_image = self.Z_images[idx % self.Z_len]
        H_image = self.H_images[idx % self.H_len]

        Z_path = os.path.join(self.root_Z, Z_image)
        H_path = os.path.join(self.root_H, H_image)

        Z_image = np.array(Image.open(Z_path).convert("RGB"))
        H_image = np.array(Image.open(H_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=Z_image, image0=H_image)
            Z_image = augmentations["image"]
            H_image = augmentations["image0"]

        return Z_image, H_image
