import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetProcessor(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        super(DatasetProcessor, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir,
            self.images[index]
            .replace(".jpg", "_segmentation.png")
            .replace(".jpeg", "_segmentation.png"),
        )

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augemantations = self.transform(image=image, mask=mask)
            image = augemantations["image"]
            mask = augemantations["mask"]
        return image, mask


train_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


def get_data_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform=train_transform,
    val_transform=val_transform,
):

    train_ds = DatasetProcessor(
        image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = DatasetProcessor(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
