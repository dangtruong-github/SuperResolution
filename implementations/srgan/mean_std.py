import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def GetMeanStd(root, batch_size):
    # Define the transform to convert images to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the dataset
    dataset = ImageDataset(folder=root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize variables to store the sums and squared sums
    mean = torch.zeros(3).to(device=device)
    std = torch.zeros(3).to(device=device)
    n_images = 0

    # Iterate over the dataset
    for images in loader:
        images = images.to(device=device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_images += batch_samples

    mean /= n_images
    std /= n_images

    mean = mean.to(device="cpu")
    std = std.to(device="cpu")

    return mean, std
