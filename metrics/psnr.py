import torch
import torch.nn.functional as F
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr

from metrics.utils import reverse_transform_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
valid_modes = ["lr", "hr"]

def calculate_psnr_batch(
    img1_batch: torch.Tensor,
    img2_batch: torch.Tensor,
    mean_std,
    img_size=(256, 256),
    batch: bool=True,
    mode: str="hr",
    scale_factor: int=4
) -> np.ndarray:
    if mode not in valid_modes:
        raise ValueError(f"mode={mode} invalid. Try one of {valid_modes}")

    batch_size = img1_batch.shape[0] if batch else 1

    img1_batch = img1_batch.to(device=device)
    img2_batch = img2_batch.to(device=device)

    if batch is False:
        img1_batch = img1_batch.unsqueeze(0)
        img2_batch = img2_batch.unsqueeze(0)

    if mode == "lr":
        img1_batch = F.interpolate(img1_batch, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
        img2_batch = F.interpolate(img2_batch, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
    
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
