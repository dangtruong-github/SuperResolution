import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models import GeneratorResNet, Discriminator, FeatureExtractor
from datasets import CreateLoader
from metrics import *

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Denormalization function
def denormalize(tensor,
                mean=torch.tensor(np.array([0.485, 0.456, 0.406])),
                std=torch.tensor(np.array([0.229, 0.224, 0.225]))):
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


def GetModel(gen_path=None, dis_path=None):
    # Model preparation
    generator = GeneratorResNet()

    if gen_path is not None:
        checkpoint_g = torch.load(gen_path, map_location="cpu", weights_only=True)
        generator.load_state_dict(checkpoint_g)

    discriminator = Discriminator(input_shape=(3, 256, 256))

    if dis_path is not None:
        checkpoint_d = torch.load(dis_path, map_location="cpu", weights_only=True)
        discriminator.load_state_dict(checkpoint_d)

    discriminator.eval()

    feature_extractor = FeatureExtractor()

    return generator, discriminator, feature_extractor


def GetLossFuncs():
    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    return criterion_GAN, criterion_content


def ForwardLoop(generator, discriminator, feature_extractor, imgs_lr, imgs_hr):
    criterion_GAN, criterion_content = GetLossFuncs()
    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

    # ------------------
    #  Train Generators
    # ------------------

    # Generate a high resolution image from low resolution input
    gen_hr = generator(imgs_lr)

    # Adversarial loss
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


def GenerateSomeSample():
    GEN_PATH = "saved_models/generator_5.pth"
    DIS_PATH = "saved_models/discriminator_5.pth"
    DATA_PATH = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba"

    loader = CreateLoader(root=DATA_PATH, hr_shape=(256, 256))
    dataset = loader.dataset
    img_dict = dataset.__getitem__(0)
    torch_lr = img_dict["lr"]
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
    
    torch_gen_hr, _ = ForwardLoop(
        generator, discriminator, feature_extractor, torch_lr, torch_hr
    )

    gen_hr = torch_gen_hr.squeeze(dim=0).detach()
    gen_hr_orig = reverse_transform(gen_hr, (256, 256))
    gen_hr_np = np.array(gen_hr_orig)
    plt.imshow(gen_hr_np)
    plt.savefig("gen_hr.png")
    plt.clf()

def EvaluateData(gen_path, dis_path, data_path, hr_shape):
    total_loss_G = 0.0
    total_loss_D = 0.0
    total_psnr_score = 0.0
    total_ssim_score = 0.0
    total_lpips_score = 0.0
    total_lr_psnr_score = 0.0

    # Model preparation
    loader = CreateLoader(root=data_path, hr_shape=hr_shape)
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
            psnr_score = calculate_psnr_batch(imgs_hr, gen_hr)
            ssim_score = calculate_ssim_batch(imgs_hr, gen_hr)
            lpips_score = calculate_lpips_batch(imgs_hr, gen_hr)
            lr_psnr_score = calculate_lr_psnr_batch(imgs_hr, gen_hr)

            cur_batch_size = imgs_hr.shape[0]

            print(f"PSNR score: {(psnr_score.sum()) / cur_batch_size}")
            print(f"SSIM score: {ssim_score.sum() / cur_batch_size}")
            print(f"LPIPS score: {lpips_score.sum() / cur_batch_size}")
            print(f"LR_PSNR score: {lr_psnr_score.sum() / cur_batch_size}")

            total_psnr_score += psnr_score.sum()
            total_ssim_score += ssim_score.sum()
            total_lpips_score += lpips_score.sum()
            total_lr_psnr_score += lr_psnr_score.sum()
    
    data_size = 320
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
    GEN_PATH = "saved_models/generator_5.pth"
    DIS_PATH = "saved_models/discriminator_5.pth"
    DATA_PATH = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/img_align_celeba"
    GenerateSomeSample()
    EvaluateData(GEN_PATH, DIS_PATH, DATA_PATH, (256, 256))