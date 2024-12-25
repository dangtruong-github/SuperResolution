import torch
from torchvision import transforms
from PIL import Image

import glob
import os

hr_height, hr_width = 256, 256
lr_height, lr_width = 64, 64

valid_modes = ["to_hr", "to_lr"]

def transform_img(image_path, save_path, mode="to_hr"):
    
    if mode not in valid_modes:
        raise ValueError(f"{mode} invalid. Try {valid_modes}")
    # Load your image
    image = Image.open(image_path)

    size_height, size_width = hr_height, hr_width
    if mode == "to_lr":
        size_height, size_width = lr_height, lr_width

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((size_height, size_width), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])

    # Apply the transform to the image
    resized_image = transform(image)

    # Save or display the resized image
    resized_image.save(save_path)

    print(f"Save new image success in {save_path}")


if __name__ == "__main__":
    """folder_hr = "../data/celeba_hq/hr"
    for file_hr in os.listdir(folder_hr):
        file_total_hr = os.path.join(folder_hr, file_hr)
        file_total_lr = file_total_hr.replace("/hr/", "/lr/")

        if os.path.exists(file_total_lr):
            continue
        transform_img(file_total_hr, file_total_lr, mode="to_lr")"""
    
    folder_orig = "../data/celeba/img_align_celeba"
    folder_hr = folder_orig.replace("/img_align_celeba", "/hr")
    folder_lr = folder_orig.replace("/img_align_celeba", "/lr")

    os.makedirs(folder_hr, exist_ok=True)
    os.makedirs(folder_lr, exist_ok=True)

    for file_orig in os.listdir(folder_orig):
        file_total_orig = os.path.join(folder_orig, file_orig)
        file_total_hr = file_total_orig.replace("/img_align_celeba/", "/hr/")
        file_total_lr = file_total_orig.replace("/img_align_celeba/", "/lr/")

        if os.path.exists(file_total_hr):
            if os.path.exists(file_total_lr):
                continue
            transform_img(file_total_hr, file_total_lr, mode="to_lr")

        transform_img(file_total_orig, file_total_hr, mode="to_hr")
        
        transform_img(file_total_hr, file_total_lr, mode="to_lr")

        # break
