import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

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
    # Normalization parameters for pre-trained PyTorch models
    if ((root == "../../data/celeba_hq_256") 
        or (root == "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256")):
        # test set
        mean = torch.tensor([0.5171, 0.4167, 0.3635])
        std = torch.tensor([0.3009, 0.2723, 0.2674])
        return mean, std
    elif ((root == "../../data/img_align_celeba/train") 
        or (root == "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/train")):
        # train set
        mean = torch.tensor([0.5060, 0.4253, 0.3827])
        std = torch.tensor([0.3101, 0.2898, 0.2891])
        return mean, std
    elif ((root == "../../data/img_align_celeba/val") 
        or (root == "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba/val")):
        mean = torch.tensor([0.5069, 0.4263, 0.3840])
        std = torch.tensor([0.3100, 0.2898, 0.2894])
        return mean, std

    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    mean = torch.tensor(mean)
    std = torch.tensor(std)
    
    # train
    mean = torch.tensor([0.5060, 0.4253, 0.3827])
    std = torch.tensor([0.3101, 0.2898, 0.2891])

    # test
    (tensor([0.5171, 0.4167, 0.3635]), tensor([0.3009, 0.2723, 0.2674]))

    return (mean, std)
    """

    img_height = 256
    img_width = 256

    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )

    # Load the dataset
    dataset = ImageDataset(folder=root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize variables to store the sums and squared sums
    n_images = len(dataset)
    print(f"Length of dataset: {n_images}")
    
    psum = torch.tensor([0.0, 0.0, 0.0]).to(device=device)
    psum_sq = torch.tensor([0.0, 0.0, 0.0]).to(device=device)

    # Iterate over the dataset
    for index, images in enumerate(loader):
        images = images.to(device=device)

        psum    += images.sum(axis        = [0, 2, 3])
        psum_sq += (images ** 2).sum(axis = [0, 2, 3])

        if (index+1) % int(10000 / batch_size) == 0:
            print(f"Finish batch: {index+1}/{len(loader)}")
    count = n_images * img_height * img_width

    # mean and STD
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    mean = total_mean.to(device="cpu")
    std = total_std.to(device="cpu")

    return mean, std
