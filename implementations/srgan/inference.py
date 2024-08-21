import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models import GetModel, GetLossFuncs
from datasets import CreateLoader
from metrics import *
from mean_std import GetMeanStd

import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Denormalization function
def denormalize(tensor,
                mean=torch.tensor(np.array([0.485, 0.456, 0.406])),
                std=torch.tensor(np.array([0.229, 0.224, 0.225]))):
    tensor = tensor.to(device=device)
    mean = mean.to(device=device)
    std = std.to(device=device)

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# Reversing the transformations
def reverse_transform(tensor, original_size):
    # Denormalize the tensor
    tensor = denormalize(tensor, mean, std)
    
    # Convert the tensor back to a PIL image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    
    # Resize back to the original size
    image = image.resize(original_size, Image.BICUBIC)
    
    return image


def ForwardLoop(generator, discriminator, feature_extractor, imgs_lr, imgs_hr):
    criterion_GAN, criterion_content = GetLossFuncs()
    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

    # ------------------
    #  Train Generators
    # ------------------

    imgs_lr = imgs_lr.to(device=device)
    imgs_hr = imgs_hr.to(device=device)

    # Generate a high resolution image from low resolution input
    #print(next(generator.parameters()).is_cuda)
    gen_hr = generator(imgs_lr)

    # Adversarial loss
    #print(next(discriminator.parameters()).is_cuda)
    loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

    # Content loss
    gen_features = feature_extractor(gen_hr)
    real_features = feature_extractor(imgs_hr)
    loss_content = criterion_content(gen_features, real_features.detach())

    # Total loss
    loss_G = loss_content + 1e-3 * loss_GAN

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Loss of real and fake images
    loss_real = criterion_GAN(discriminator(imgs_hr), valid)
    loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

    # Total loss
    loss_D = (loss_real + loss_fake) / 2

    return gen_hr, (loss_G, loss_D)


def GenerateSomeSample(gen_path, dis_path, data_path, hr_shape):
    GEN_PATH = gen_path
    DIS_PATH = dis_path
    DATA_PATH = data_path

    loader = CreateLoader(root=DATA_PATH, hr_shape=hr_shape)
    dataset = loader.dataset
    img_dict = dataset.__getitem__(0)
    torch_lr = img_dict["lr"].to(device=device)
    torch_hr = img_dict["hr"]
    img_lr = reverse_transform(torch_lr, (64, 64))
    img_hr = reverse_transform(torch_hr, (256, 256))
    img_lr_np = np.array(img_lr)
    img_hr_np = np.array(img_hr)
    #img_lr_pil = Image.fromarray(img_lr_np)
    #print(img_lr_pil)

    plt.imshow(img_lr_np)
    plt.savefig("img_lr.png")
    plt.clf()
    plt.imshow(img_hr_np)
    plt.savefig("img_hr.png")
    plt.clf()

    # Model preparation
    generator, discriminator, feature_extractor = GetModel(GEN_PATH, DIS_PATH)

    torch_lr = torch_lr.unsqueeze(0)
    torch_hr = torch_hr.unsqueeze(0)
    
    with torch.no_grad():
        torch_gen_hr, _ = ForwardLoop(
            generator, discriminator, feature_extractor, torch_lr, torch_hr
        )

    gen_hr = torch_gen_hr.squeeze(dim=0).detach()
    gen_hr_orig = reverse_transform(gen_hr, (256, 256))
    gen_hr_np = np.array(gen_hr_orig)
    plt.imshow(gen_hr_np)
    plt.savefig("gen_hr.png")
    plt.clf()

def EvaluateData(gen_path, dis_path, data_path, hr_shape, batch_size=32):
    total_loss_G = 0.0
    total_loss_D = 0.0
    total_psnr_score = 0.0
    total_ssim_score = 0.0
    total_lpips_score = 0.0
    total_lr_psnr_score = 0.0

    mean_std = GetMeanStd(root=data_path, batch_size=batch_size)
    print(mean_std)

    return

    # Model preparation
    loader = CreateLoader(root=data_path, hr_shape=hr_shape, batch_size=batch_size)
    generator, discriminator, feature_extractor = GetModel(gen_path, dis_path)

    with torch.no_grad():
        for _, imgs in enumerate(loader):
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            gen_hr, (loss_G, loss_D) = ForwardLoop(
                generator, discriminator, feature_extractor, imgs_lr, imgs_hr
            )

            total_loss_G += loss_G
            total_loss_D += loss_D

            # Evaluate
            psnr_score = calculate_psnr_batch(imgs_hr, gen_hr, mean_std)
            ssim_score = calculate_ssim_batch(imgs_hr, gen_hr, mean_std)
            lpips_score = calculate_lpips_batch(imgs_hr, gen_hr, mean_std)
            lr_psnr_score = calculate_lr_psnr_batch(imgs_hr, gen_hr, mean_std)

            cur_batch_size = imgs_hr.shape[0]

            print(psnr_score)
            print(ssim_score)
            print(lpips_score)
            print(lr_psnr_score)

            print(f"PSNR score: {(psnr_score.sum()) / cur_batch_size}")
            print(f"SSIM score: {ssim_score.sum() / cur_batch_size}")
            print(f"LPIPS score: {lpips_score.sum() / cur_batch_size}")
            print(f"LR_PSNR score: {lr_psnr_score.sum() / cur_batch_size}")

            total_psnr_score += psnr_score.sum()
            total_ssim_score += ssim_score.sum()
            total_lpips_score += lpips_score.sum()
            total_lr_psnr_score += lr_psnr_score.sum()
    
    data_size = len(loader.dataset)
    avg_psnr_score = total_psnr_score / data_size
    avg_ssim_score = total_ssim_score / data_size
    avg_lpips_score = total_lpips_score / data_size
    avg_lr_psnr_score = total_lr_psnr_score / data_size

    print(f"Total loss G: {total_loss_G}")
    print(f"Total loss D: {total_loss_D}")
    print(f"Average PSNR score: {avg_psnr_score}")
    print(f"Average SSIM score: {avg_ssim_score}")
    print(f"Average LPIPS score: {avg_lpips_score}")
    print(f"Average LR_PSNR score: {avg_lr_psnr_score}")

if __name__ == "__main__":
    GEN_PATH = "saved_models/generator_19.pth"
    DIS_PATH = "saved_models/discriminator_19.pth"
    DATA_PATH = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba"
    GenerateSomeSample(GEN_PATH, DIS_PATH, DATA_PATH, (256, 256))
    EvaluateData(GEN_PATH, DIS_PATH, DATA_PATH, (256, 256))