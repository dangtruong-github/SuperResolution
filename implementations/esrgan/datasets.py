import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    def __init__(self, root): #, hr_shape=(256, 256)):
        #hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                #transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        """
        self.hr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        """

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        path_hr = self.files[index % len(self.files)]
        path_lr = path_hr.replace("/hr/", "/lr/")
        img_lr = Image.open(path_lr)
        img_hr = Image.open(path_hr)
        img_torch_lr = self.transform(img_lr)
        img_torch_hr = self.transform(img_hr)

        return {"lr": img_torch_lr, "hr": img_torch_hr}

    def __len__(self):
        return len(self.files)
