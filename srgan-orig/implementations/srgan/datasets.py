import glob
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os

from mean_std import GetMeanStd


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        mean, std = GetMeanStd(root, batch_size=64)

        print(mean, std)

        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        """
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        """

        self.transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = os.listdir(os.path.join(root, "hr"))
        self.mean = mean
        self.std = std
        self.root = root

    def __getitem__(self, index):
        path_lr = os.path.join(self.root, "lr", self.files[index % len(self.files)])
        path_hr = os.path.join(self.root, "hr", self.files[index % len(self.files)])
        img_lr = Image.open(path_lr)
        img_hr = Image.open(path_hr)
        img_lr = self.transform_img(img_lr)
        img_hr = self.transform_img(img_hr)

        return {"lr": img_lr, "hr": img_hr, "files": self.files[index % len(self.files)]}

    def __len__(self):
        return len(self.files)


def CreateLoader(root, hr_shape, batch_size=32, shuffle=False):
    dataset = ImageDataset(root, hr_shape)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
