import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Denormalization function that handles batch processing
def denormalize_batch(tensors, mean, std):
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    tensors = tensors * std + mean
    return tensors


# Reversing the transformations for a batch
def reverse_transform_batch(
    tensors,
    original_size,
    mean_std=None,
    return_tensor=True,
    return_type="cpu"
):
    # Denormalize the batch of tensors
    mean, std = mean_std
    tensors = denormalize_batch(tensors, mean, std)
    
    # Convert each tensor in the batch to a PIL image and resize
    to_pil = transforms.ToPILImage()
    images = [to_pil(tensor).resize(original_size, Image.BICUBIC) for tensor in tensors]

    if return_tensor:
        images = [np.asarray(imagen) for imagen in images]
        images = np.array(images)
        images = torch.tensor(images).to(device=return_type)
    
    return images