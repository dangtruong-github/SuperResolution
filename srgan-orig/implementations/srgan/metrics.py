import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Assume mean and std are the same values used in your normalization step
mean = torch.tensor(mean)
std = torch.tensor(std)

loss_fn = lpips.LPIPS(net='alex').to(device="cpu")


# Denormalization function that handles batch processing
def denormalize_batch(tensors, mean, std):
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    tensors = tensors * std + mean
    return tensors


# Reversing the transformations for a batch
def reverse_transform_batch(tensors, original_size, mean_std=None, return_tensor=True):
    # Denormalize the batch of tensors
    mean, std = mean_std
    tensors = denormalize_batch(tensors, mean, std)
    
    # Convert each tensor in the batch to a PIL image and resize
    to_pil = transforms.ToPILImage()
    images = [to_pil(tensor).resize(original_size, Image.BICUBIC) for tensor in tensors]

    if return_tensor:
        images = [np.asarray(imagen) for imagen in images]
        images = np.array(images)
        images = torch.tensor(images)
    
    return images


def calculate_psnr_batch(img1_batch: torch.Tensor, img2_batch: torch.Tensor,
                         mean_std, img_size=(256, 256),
                         batch: bool=True) -> np.ndarray:
    batch_size = img1_batch.shape[0] if batch else 1

    img1_batch = img1_batch.to(device="cpu")
    img2_batch = img2_batch.to(device="cpu")

    if batch is False:
        img1_batch = img1_batch.unsqueeze(0)
        img2_batch = img2_batch.unsqueeze(0)
    
    imgs1_np = reverse_transform_batch(img1_batch, original_size=img_size, mean_std=mean_std)
    imgs2_np = reverse_transform_batch(img2_batch, original_size=img_size, mean_std=mean_std)

    imgs1_np = np.array(imgs1_np)
    imgs2_np = np.array(imgs2_np)

    psnr_values = np.zeros(batch_size)
    for i in range(batch_size):
        psnr_values[i] = psnr(imgs1_np[i], imgs2_np[i])

    if batch:
        return psnr_values
    else:
        return psnr_values[0]


def calculate_ssim_batch(img1_batch: np.ndarray, img2_batch: np.ndarray,
                         mean_std, img_size=(256, 256),
                         batch: bool=True) -> np.ndarray:
    batch_size = img1_batch.shape[0] if batch else 1

    img1_batch = img1_batch.to(device="cpu")
    img2_batch = img2_batch.to(device="cpu")

    if batch is False:
        img1_batch = img1_batch.unsqueeze(0)
        img2_batch = img2_batch.unsqueeze(0)
    
    imgs1_np = reverse_transform_batch(img1_batch, original_size=img_size, mean_std=mean_std)
    imgs2_np = reverse_transform_batch(img2_batch, original_size=img_size, mean_std=mean_std)

    imgs1_np = np.array(imgs1_np)
    imgs2_np = np.array(imgs2_np)

    ssim_values = np.zeros(batch_size)
    for i in range(batch_size):
        ssim_values[i] = ssim(imgs1_np[i], imgs2_np[i],
                              multichannel=True, channel_axis=2)

    if batch:
        return ssim_values
    else:
        return ssim_values[0]


def calculate_lr_psnr_batch(
    img1_batch: torch.Tensor,
    img2_batch: torch.Tensor,
    mean_std,
    img_size=(256, 256),
    scale_factor: int=4
) -> np.ndarray:
    # Downsample the images
    img1_batch = img1_batch.to(device="cpu")
    img2_batch = img2_batch.to(device="cpu")

    img1_lr_batch = F.interpolate(img1_batch, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
    img2_lr_batch = F.interpolate(img2_batch, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)

    img1_lr_batch = reverse_transform_batch(img1_lr_batch, original_size=img_size, mean_std=mean_std)
    img2_lr_batch = reverse_transform_batch(img2_lr_batch, original_size=img_size, mean_std=mean_std)

    batch_size = img1_lr_batch.size(0)
    psnr_values = np.zeros(batch_size)
    
    for i in range(batch_size):
        img1_lr = img1_lr_batch[i].squeeze().cpu().numpy()
        img2_lr = img2_lr_batch[i].squeeze().cpu().numpy()
        psnr_values[i] = psnr(img1_lr, img2_lr)
    
    return psnr_values


def calculate_lpips_batch(
    img1_batch: torch.Tensor,
    img2_batch: torch.Tensor,
    mean_std,
    img_size=(256, 256)
) -> torch.Tensor:
    global loss_fn
    
    img1_batch = img1_batch.to(device="cpu")
    img2_batch = img2_batch.to(device="cpu")

    img1_batch = reverse_transform_batch(
        img1_batch, original_size=img_size, mean_std=mean_std, return_tensor=True
    )
    img2_batch = reverse_transform_batch(
        img2_batch, original_size=img_size, mean_std=mean_std, return_tensor=True
    )

    img1_batch = torch.moveaxis(img1_batch, -1, 1)
    img2_batch = torch.moveaxis(img2_batch, -1, 1)
    
    # LPIPS expects images in the range [-1, 1]
    img1_batch = (img1_batch - 0.5) * 2
    img2_batch = (img2_batch - 0.5) * 2
    
    lpips_values = torch.zeros(img1_batch.size(0)).to(device="cpu")
    
    for i in range(img1_batch.size(0)):
        lpips_values[i] = loss_fn(img1_batch[i:i+1], img2_batch[i:i+1])
    
    return lpips_values
