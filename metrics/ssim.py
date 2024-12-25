import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim

from metrics.utils import reverse_transform_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_ssim_batch(img1_batch: np.ndarray, img2_batch: np.ndarray,
                         mean_std, img_size=(256, 256),
                         batch: bool=True) -> np.ndarray:
    batch_size = img1_batch.shape[0] if batch else 1

    img1_batch = img1_batch.to(device=device)
    img2_batch = img2_batch.to(device=device)

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
