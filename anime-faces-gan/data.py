from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

batch_size = 64
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform = transforms.Compose(
    [
        transforms.CenterCrop(64),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)


def denorm(image_tensors):
    """
    Denormalize the image tensors.
    """
    return image_tensors * stats[1][0] + stats[0][0]


class AnimeDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.frame = pd.read_csv(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        image = Image.open(image_name)
        image = self.transform(image)
        return image


if __name__ == "__main__":
    dataset = AnimeDataset(root_dir="./data.csv", transform=transform)
    trainloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
