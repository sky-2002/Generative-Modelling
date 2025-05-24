from torchvision import transforms
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_EPOCHS = 2
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0
DATA_ROOT = "/home/aakash/personal/GANS/cycleGAN/monet2photo"
IMAGES_ROOT = "/home/aakash/personal/GANS/cycleGAN/saved_images"


def save_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")

    torch.save(
        model.state_dict(),
        filename,
    )


def save_image(image, filename="image.png"):
    image = image * 0.5 + 0.5
    image = transforms.ToPILImage()(image)
    image.save(filename)
