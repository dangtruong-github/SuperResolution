"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import re

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from implementations.esrgan.models import *
from implementations.esrgan.forward_pass import forward_pass
from datasets.loader import *

from metrics.psnr import calculate_psnr_batch
from metrics.ssim import calculate_ssim_batch
from metrics.lpips import calculate_lpips_batch
from datasets.get_data import get_data

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_file_format(file_name):
    pattern = re.compile(r"^generator_\d+\.pth$")
    match = pattern.match(file_name)
    if match:
        return int(file_name[:-4].split("_")[1])
    return None

def esrgan_train(opt):
    print(opt.test_mode)
    hr_shape = (opt.hr_height, opt.hr_width)

    save_path = os.path.join("./saved_models", "esrgan", opt.save_path)
    model_path = os.path.join(save_path, "models")
    img_path = os.path.join(save_path, "images")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    
    best_loss_G = 1e6
    best_loss_D = 1e6
    if opt.epoch < 0:
        best_epoch = 0
        for model_file in os.listdir("{}".format(model_path)):
            new_epoch = check_file_format(model_file)

            if new_epoch is None:
                continue

            best_epoch = max(best_epoch, new_epoch)

        opt.epoch = best_epoch
    if opt.epoch > 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(
            "{}/generator_{}.pth".format(model_path, opt.epoch),
            map_location=device
        ))
        discriminator.load_state_dict(torch.load(
            "{}/discriminator_{}.pth".format(model_path, opt.epoch),
            map_location=device
        ))

        with open(os.path.join(model_path, "best_loss.txt"), "r") as f:
            for line in f.readlines():
                if "Best generator loss:" in line:
                    # Extract the value after the colon and strip any whitespace or newline characters
                    best_loss_G = float(line.split(":")[1].strip())
                elif "Best discriminator loss:" in line:
                    # Extract the value after the colon and strip any whitespace or newline characters
                    best_loss_D = float(line.split(":")[1].strip())

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    (train_loader, val_loader), (mean_torch, std_torch) = get_data(opt)

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epoch+1, opt.n_epochs):
        if epoch >= opt.epoch+3 and opt.test_mode == True:
            break

        train_loss_G = 0.0
        train_loss_D = 0.0
        train_loss_GAN = 0.0
        train_loss_content = 0.0
        train_loss_pixel = 0.0

        val_loss_G = 0.0
        val_loss_D = 0.0
        val_loss_GAN = 0.0
        val_loss_content = 0.0
        val_loss_pixel = 0.0

        train_hr_psnr = []
        train_lr_psnr = []
        train_ssim = []
        train_lpips = []

        val_hr_psnr = []
        val_lr_psnr = []
        val_ssim = []
        val_lpips = []

        for i, imgs in enumerate(train_loader):
            if i >= 2 and opt.test_mode == True:
                break

            batches_done = (epoch - 1) * len(train_loader) + i
            print(batches_done)

            if batches_done < opt.warmup_batches:
                loss_pixel, models_new = forward_pass(
                    generator, discriminator, imgs,
                    type_forward="train",
                    optimizer_G=optimizer_G, optimizer_D=optimizer_D,
                    loss_gan_rate=opt.lambda_adv, loss_pixel_rate=opt.lambda_pixel,
                    warm_up_batch=True
                )
                generator, discriminator, optimizer_G, optimizer_D = models_new
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(train_loader), loss_pixel.item())
                )
                continue

            loss_G, loss_D, loss_each, models_new, gen_hr = forward_pass(
                generator, discriminator, imgs,
                type_forward="train",
                optimizer_G=optimizer_G, optimizer_D=optimizer_D,
                loss_gan_rate=opt.lambda_adv, loss_pixel_rate=opt.lambda_pixel,
                warm_up_batch=False
            )

            loss_GAN, loss_content, loss_pixel = loss_each
            generator, discriminator, optimizer_G, optimizer_D = models_new

            # Evaluate
            imgs_hr = imgs["hr"].to(device=device)
            
            hr_psnr_values = calculate_psnr_batch(
                imgs_hr, gen_hr, mean_std=(mean_torch, std_torch)
            )
            lr_psnr_values = calculate_psnr_batch(
                imgs_hr, gen_hr, mode="lr", mean_std=(mean_torch, std_torch)
            )
            ssim_values = calculate_ssim_batch(
                imgs_hr, gen_hr, mean_std=(mean_torch, std_torch)
            )
            lpips_values = calculate_lpips_batch(imgs_hr, gen_hr, mean_std=(mean_torch, std_torch)
            )

            train_hr_psnr.append(hr_psnr_values)
            train_lr_psnr.append(lr_psnr_values)
            train_ssim.append(ssim_values)
            train_lpips.append(lpips_values)

            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()
            train_loss_GAN += loss_GAN.item()
            train_loss_content += loss_content.item()
            train_loss_pixel += loss_pixel.item()

            # --------------
            #  Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(train_loader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )
            

            if batches_done % opt.sample_interval == 0:
                imgs_lr = imgs["lr"].to(device=device)
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1), mean_torch, std_torch)
                save_image(img_grid, "{}/{}.png".format(img_path, batches_done), nrow=1, normalize=False)

        train_loss_D /= len(train_loader)
        train_loss_G /= len(train_loader)
        train_loss_GAN /= len(train_loader)
        train_loss_content /= len(train_loader)
        train_loss_pixel /= len(train_loader)

        if len(train_hr_psnr) > 0:
            train_hr_psnr = np.mean(np.concatenate(train_hr_psnr))
            train_lr_psnr = np.mean(np.concatenate(train_lr_psnr))
            train_ssim = np.mean(np.concatenate(train_ssim))
            train_lpips = np.mean(np.concatenate(train_lpips))
            
            sys.stdout.write(
                "[PSNR %f] [LR-PSNR %f] [SSIM %f] [LPIPS %f]\n"
                % (train_hr_psnr, train_lr_psnr, train_ssim, train_lpips)
            )

        sys.stdout.write(
            "Total Training Stats: [Epoch %d/%d] [D loss: %f] [GAN loss: %f] [content loss: %f] [pixel loss: %f] [G loss: %f]\n"
            % (epoch, opt.n_epochs, train_loss_D, train_loss_GAN, train_loss_content, train_loss_pixel, train_loss_G)
        )
        
        with torch.no_grad():
            for i, imgs in enumerate(val_loader):
                if i >= 2 and opt.test_mode == True:
                    break

                loss_G, loss_D, loss_each, _, gen_hr = forward_pass(
                    generator, discriminator, imgs,
                    type_forward="eval",
                    loss_gan_rate=opt.lambda_adv, loss_pixel_rate=opt.lambda_pixel,
                    warm_up_batch=False
                )

                loss_GAN, loss_content, loss_pixel = loss_each
                # Evaluate
                imgs_hr = imgs["hr"].to(device=device)
                
                hr_psnr_values = calculate_psnr_batch(
                    imgs_hr, gen_hr, mean_std=(mean_torch, std_torch))
                lr_psnr_values = calculate_psnr_batch(
                    imgs_hr, gen_hr, mode="lr", mean_std=(mean_torch, std_torch)
                )
                ssim_values = calculate_ssim_batch(
                    imgs_hr, gen_hr, mean_std=(mean_torch, std_torch)
                )
                lpips_values = calculate_lpips_batch(
                    imgs_hr, gen_hr, mean_std=(mean_torch, std_torch)
                )

                val_hr_psnr.append(hr_psnr_values)
                val_lr_psnr.append(lr_psnr_values)
                val_ssim.append(ssim_values)
                val_lpips.append(lpips_values)

                val_loss_G += loss_G.item()
                val_loss_D += loss_D.item()
                val_loss_GAN += loss_GAN.item()
                val_loss_content += loss_content.item()
                val_loss_pixel += loss_pixel.item()

        val_loss_D /= len(train_loader)
        val_loss_G /= len(train_loader)
        val_loss_GAN /= len(train_loader)
        val_loss_content /= len(train_loader)
        val_loss_pixel /= len(train_loader)

        val_hr_psnr = np.mean(np.concatenate(val_hr_psnr))
        val_lr_psnr = np.mean(np.concatenate(val_lr_psnr))
        val_ssim = np.mean(np.concatenate(val_ssim))
        val_lpips = np.mean(np.concatenate(val_lpips))

        
        sys.stdout.write(
            "Total Training Stats: [Epoch %d/%d] [D loss: %f] [GAN loss: %f] [content loss: %f] [pixel loss: %f] [G loss: %f]\n"
            % (epoch, opt.n_epochs, val_loss_D, val_loss_GAN, val_loss_content, val_loss_pixel, val_loss_G)
        )
        sys.stdout.write(
            "[PSNR %f] [LR-PSNR %f] [SSIM %f] [LPIPS %f]\n"
            % (val_hr_psnr, val_lr_psnr, val_ssim, val_lpips)
        )


        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "{}/generator_{}.pth".format(model_path, epoch))
            torch.save(discriminator.state_dict(), "{}/discriminator_{}.pth".format(model_path, epoch))

        if val_loss_G < best_loss_G and val_loss_D < best_loss_D:
            # Save model checkpoints
            torch.save(generator.state_dict(), "{}/generator_best.pth".format(model_path))
            torch.save(discriminator.state_dict(), "{}/discriminator_best.pth".format(model_path))
            
            with open(os.path.join(model_path, "best_loss.txt"), "w") as f:
                f.write(f"Best generator loss: {val_loss_G}\n")
                f.write(f"Best discriminator loss: {val_loss_D}\n")
                f.write(f"Best epoch: {epoch}")
            
            best_loss_G = val_loss_G
            best_loss_D = val_loss_D

            sys.stdout.write(
                "Save best checkpoint: Epoch %d/%d\n"
                % (epoch, opt.n_epochs)
            )
        sys.stdout.write(
            "--------------Finish Epoch %d/%d-----------------------------\n"
            % (epoch, opt.n_epochs)
        )   

