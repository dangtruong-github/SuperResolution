import numpy as np

import os
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

import time

class CustomImageDataset(Dataset): 
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform: 
            image = self.transform(image)
        
        return image

def calculate_mean_std(image_folder, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

    dataset = CustomImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = torch.zeros(3).to('cuda')
    std = torch.zeros(3).to('cuda')
    total_images = 0

    for images in dataloader:
        images = images.to('cuda')
        batch_samples = images.size(0)  # Batch size (the last batch can have smaller size)
        total_images += batch_samples

        # Compute mean and std for each batch
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples

        # break

    mean /= total_images
    std /= total_images

    return mean.cpu().numpy(), std.cpu().numpy()

# Example usage
dataset_type = "celeba_hq"
image_folder = '../data/{}/hr'.format(dataset_type)

start_time = time.time()

mean, std = calculate_mean_std(image_folder, batch_size=8192)

mean_path = '../data/{}/mean.npy'.format(dataset_type)
std_path = '../data/{}/std.npy'.format(dataset_type)

print(f"Mean: {mean}")
print(f"Std: {std}")

np.save(mean_path, mean)
print(f"Save mean of {dataset_type} success in {mean_path}")
np.save(std_path, std)
print(f"Save std of {dataset_type} success in {std_path}")

# Code to measure
end_time = time.time()

execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

