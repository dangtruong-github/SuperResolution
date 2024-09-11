import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

from models import GetModel, GetLossFuncs
from datasets import CreateLoader, ImageDataset
from metrics import *
from mean_std import GetMeanStd

import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="super-resolution",
)

print(device)

# Denormalization function
def denormalize(tensor,
                mean=torch.tensor(np.array([0.485, 0.456, 0.406])),
                std=torch.tensor(np.array([0.229, 0.224, 0.225]))):
    tensor = tensor.to(device=device)
    mean = mean.to(device=device)
    std = std.to(device=device)

    print(tensor[0, 0, 0])
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    print(tensor[0, 0, 0])
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


def PrintImage(torch_img, location):
    mean = torch.tensor([0.5060, 0.4253, 0.3827])
    std = torch.tensor([0.3101, 0.2898, 0.2891])

    print("-------------------")
    print(torch_img[0, 0, 0])
    tensor = denormalize(torch_img, mean, std)
    print(tensor[0, 0, 0])
    print(tensor.shape)
    print("-------------------")

    # Convert the tensor back to a PIL image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    
    # Resize back to the original size
    image = image.resize((256, 256), Image.BICUBIC)
    
    img_hr_np = np.array(image)
    #img_lr_pil = Image.fromarray(img_lr_np)
    #print(img_lr_pil)

    plt.imshow(img_hr_np)
    plt.savefig(location)
    plt.clf()


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


def PrintWorstImages(worst_imgs, root, hr_shape, gen_path, dis_path, folder_save):
    custom_img_set = ImageDataset(worst_imgs, root, hr_shape)

    hr_height, hr_width = hr_shape
    
    generator, discriminator, feature_extractor = GetModel(gen_path, dis_path)
    generator.eval()
    discriminator.eval()

    lr_height = (hr_height // 4)
    lr_width = (hr_width // 4)

    img_hr_np_total = np.zeros((hr_height * 15, hr_width * 20, 3))
    gen_hr_np_total = np.zeros((hr_height * 15, hr_width * 20, 3))
    img_lr_np_total = np.zeros((lr_height * 15, lr_width * 20, 3))

    for id in range(len(custom_img_set)):
        img_dict = custom_img_set.__getitem__(id)
        torch_lr = img_dict["lr"].to(device=device)
        torch_hr = img_dict["hr"]
        img_hr = reverse_transform(torch_hr, hr_shape)
        img_hr_np = np.array(img_hr)
        img_lr = reverse_transform(torch_lr, (lr_height, lr_width))
        img_lr_np = np.array(img_lr)

        torch_lr = torch_lr.unsqueeze(0)
        torch_hr = torch_hr.unsqueeze(0)
        
        with torch.no_grad():
            torch_gen_hr, _ = ForwardLoop(
                generator, discriminator, feature_extractor, torch_lr, torch_hr
            )

        gen_hr = torch_gen_hr.squeeze(dim=0).detach()
        gen_hr_orig = reverse_transform(gen_hr, hr_shape)
        gen_hr_np = np.array(gen_hr_orig)

        row = int(id / 20)
        col = int(id % 20)

        print(row*hr_height)
        print((row+1)*hr_height)
        
        img_hr_np_total[int(row*hr_height):int((row+1)*hr_height),
                        int(col*hr_width):int((col+1)*hr_width)] = img_hr_np
        gen_hr_np_total[int(row*hr_height):int((row+1)*hr_height),
                        int(col*hr_width):int((col+1)*hr_width)] = gen_hr_np
        img_lr_np_total[int(row*lr_height):int((row+1)*lr_height),
                        int(col*lr_width):int((col+1)*lr_width)] = img_lr_np
        
        wandb_img_hr = wandb.Image(np.array(img_hr_np, dtype=np.int64), caption="Ground truth")
        wandb_img_lr = wandb.Image(np.array(img_lr_np, dtype=np.int64), caption="Low resolution")
        wandb_gen_hr = wandb.Image(np.array(gen_hr_np, dtype=np.int64), caption="Low resolution")

        wandb.log({
            f"Image {img_dict['files']}": [wandb_img_hr, wandb_img_lr, wandb_gen_hr]
        })
        
    img_hr_np_total /= 256
    gen_hr_np_total /= 256
    img_lr_np_total /= 256

    plt.imshow(img_hr_np_total)
    plt.savefig(os.path.join(folder_save, "img_hr_worse.png"))
    plt.clf()
    plt.imshow(gen_hr_np_total)
    plt.savefig(os.path.join(folder_save, "gen_hr_worse.png"))
    plt.clf()
    plt.imshow(img_lr_np_total)
    plt.savefig(os.path.join(folder_save, "img_lr_worse.png"))
    plt.clf()


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


def EvaluateLoader(generator, discriminator, feature_extractor, loader, mean_std, printImg=False):
    total_loss_G = 0.0
    total_loss_D = 0.0
    total_psnr_score = 0.0
    total_ssim_score = 0.0
    total_lpips_score = 0.0
    total_lr_psnr_score = 0.0

    generator.eval()
    discriminator.eval()

    worst_imgs = {}

    with torch.no_grad():
        for index, imgs in enumerate(loader):
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            imgs_files = imgs["files"]

            gen_hr, (loss_G, loss_D) = ForwardLoop(
                generator, discriminator, feature_extractor, imgs_lr, imgs_hr
            )

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

            # Evaluate
            psnr_score = calculate_psnr_batch(imgs_hr, gen_hr, mean_std)
            ssim_score = calculate_ssim_batch(imgs_hr, gen_hr, mean_std)
            lpips_score = calculate_lpips_batch(imgs_hr, gen_hr, mean_std)
            lr_psnr_score = calculate_lr_psnr_batch(imgs_hr, gen_hr, mean_std)

            cur_batch_size = imgs_hr.shape[0]

            if printImg:
                for file_id in range(len(imgs_files)):
                    worst_imgs[imgs_files[file_id]] = psnr_score[file_id]
                
                #print(worst_imgs)

            if (index+1) % int(5000 / cur_batch_size) == 0:
                print(f"Batch {index+1}/{len(loader)}")
                print(f"PSNR score: {(psnr_score.mean())}")
                print(f"SSIM score: {ssim_score.mean()}")
                print(f"LPIPS score: {lpips_score.mean()}")
                print(f"LR_PSNR score: {lr_psnr_score.mean()}")

            total_psnr_score += psnr_score.sum()
            total_ssim_score += ssim_score.sum()
            total_lpips_score += lpips_score.sum()
            total_lr_psnr_score += lr_psnr_score.sum()

    top_300_keys = None
    bot_300_keys = None
    if printImg:
        # Sort the dictionary by value in descending order and get the keys
        sorted_keys_desc = sorted(worst_imgs, key=worst_imgs.get, reverse=False)
        sorted_keys_asc = sorted(worst_imgs, key=worst_imgs.get, reverse=True)

        # Get the first 300 keys with the highest values
        bot_300_keys = sorted_keys_desc[:300]
        top_300_keys = sorted_keys_asc[:300]

        # Display the result
        print(top_300_keys)

    avg_loss_G = total_loss_G / len(loader)
    avg_loss_D = total_loss_D / len(loader)
    avg_psnr = total_psnr_score / len(loader.dataset)
    avg_ssim = total_ssim_score / len(loader.dataset)
    avg_lpips = total_lpips_score / len(loader.dataset)
    avg_lr_psnr = total_lr_psnr_score / len(loader.dataset)

    if printImg:
        return (avg_loss_G, avg_loss_D), (avg_psnr, avg_ssim, avg_lpips, avg_lr_psnr), (bot_300_keys, top_300_keys)
    
    return (avg_loss_G, avg_loss_D), (avg_psnr, avg_ssim, avg_lpips, avg_lr_psnr)


def EvaluateData(gen_path, dis_path, data_path, hr_shape, batch_size=32):
    mean_std = GetMeanStd(root=data_path, batch_size=batch_size)
    print(mean_std)

    # Model preparation
    loader = CreateLoader(root=data_path, hr_shape=hr_shape, batch_size=batch_size)
    generator, discriminator, feature_extractor = GetModel(gen_path, dis_path)

    (avg_loss_G, avg_loss_D), (avg_psnr, avg_ssim, avg_lpips, avg_lr_psnr), key_imgs = EvaluateLoader(
        generator, discriminator, feature_extractor, loader, mean_std, printImg=True
    )

    bot_300_keys, top_300_keys = key_imgs
    PrintWorstImages(top_300_keys, data_path, hr_shape, gen_path, dis_path, folder_save=os.getcwd())
    PrintWorstImages(bot_300_keys, data_path, hr_shape, gen_path, dis_path, folder_save="./images")

    print(f"Average loss G: {avg_loss_G}")
    print(f"Average loss D: {avg_loss_D}")
    print(f"Average PSNR score: {avg_psnr}")
    print(f"Average SSIM score: {avg_ssim}")
    print(f"Average LPIPS score: {avg_lpips}")
    print(f"Average LR_PSNR score: {avg_lr_psnr}")

    with open("./eval.log", "w") as f:
        f.write(f"Average loss G: {avg_loss_G}\n")
        f.write(f"Average loss D: {avg_loss_D}\n")
        f.write(f"Average PSNR score: {avg_psnr}\n")
        f.write(f"Average SSIM score: {avg_ssim}\n")
        f.write(f"Average LPIPS score: {avg_lpips}\n")
        f.write(f"Average LR_PSNR score: {avg_lr_psnr}")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    best_model_epoch = 28
    GEN_PATH = "saved_models/generator_{}.pth".format(best_model_epoch)
    DIS_PATH = "saved_models/discriminator_{}.pth".format(best_model_epoch)
    DATA_PATH = "/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256"
    #GenerateSomeSample(GEN_PATH, DIS_PATH, DATA_PATH, (256, 256))
    EvaluateData(GEN_PATH, DIS_PATH, DATA_PATH, (256, 256), batch_size=64)
    #PrintWorstImages(['/N/slate/tnn3/TruongChu/SRGAN/srgan-torch/data/celeba_hq_256/07130.jpg'],
    #                DATA_PATH, (256, 256), GEN_PATH, DIS_PATH, os.getcwd())
