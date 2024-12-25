import torch
import lpips

from metrics.utils import reverse_transform_batch

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net='alex').to(device=device)
#loss_fn = None

def calculate_lpips_batch(
    img1_batch: torch.Tensor,
    img2_batch: torch.Tensor,
    mean_std,
    img_size=(256, 256)
) -> torch.Tensor:
    global loss_fn
    
    img1_batch = img1_batch.to(device=device)
    img2_batch = img2_batch.to(device=device)

    img1_batch = reverse_transform_batch(
        img1_batch, original_size=img_size, mean_std=mean_std, return_tensor=True, return_type=device
    )
    img2_batch = reverse_transform_batch(
        img2_batch, original_size=img_size, mean_std=mean_std, return_tensor=True, return_type=device
    )

    img1_batch = torch.moveaxis(img1_batch, -1, 1)
    img2_batch = torch.moveaxis(img2_batch, -1, 1)
    
    # LPIPS expects images in the range [-1, 1]
    img1_batch = (img1_batch - 0.5) * 2
    img2_batch = (img2_batch - 0.5) * 2
    
    lpips_values = torch.zeros(img1_batch.size(0))
    
    for i in range(img1_batch.size(0)):
        lpips_values[i] = loss_fn(img1_batch[i:i+1], img2_batch[i:i+1])
    
    return lpips_values.to(device="cpu").detach()
