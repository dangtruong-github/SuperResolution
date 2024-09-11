from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os

# Define the image height and the normalization parameters
hr_height = 256  # Replace this with the height of your high-resolution image

# Define the transformation
lr_transform = transforms.Compose(
    [
        transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
    ]
)

hr_transform = transforms.Compose(
    [
        transforms.Resize((hr_height, hr_height), Image.BICUBIC),
    ]
)

# Load an image
folder_to_read = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/hr"
output_folder = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/lr"  # Replace with your desired output folder
os.makedirs(output_folder, exist_ok=True)

for image_id, image_path in enumerate(os.listdir(folder_to_read)):
    image_path_total = os.path.join(folder_to_read, image_path)
    output_path_lr = os.path.join(output_folder, image_path)
    output_path_hr = os.path.join(folder_to_read, image_path)

    if os.path.exists(output_path_lr):
        continue

    image = Image.open(image_path_total)

    # Apply the transformation to the image
    resized_lr = lr_transform(image)
    resized_hr = hr_transform(image)

    # Save the image to the specified folder
    resized_lr.save(output_path_lr)
    print(f"Image saved to {output_path_lr}")
    resized_hr.save(output_path_hr)
    print(f"Image saved to {output_path_hr}")

