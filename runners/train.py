import os
import numpy as np
import sys
import re

from torchvision.utils import save_image, make_grid

import torch
import torch.nn as nn

from implementations.srgan.models import (
    GeneratorResNet, Discriminator, FeatureExtractor
)
from implementations.srgan.forward_pass import forward_pass

from metrics.psnr import calculate_psnr_batch
from metrics.ssim import calculate_ssim_batch
from metrics.lpips import calculate_lpips_batch
from datasets.get_data import get_data

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"


def check_file_format(file_name):
    pattern = re.compile(r"^generator_\d+\.pth$")
    match = pattern.match(file_name)
    if match:
        return int(file_name[:-4].split("_")[1])
    return None


def train(opt_total):
    opt_data = opt_total.data
    opt_model = opt_total.model

    hr_shape = (opt_data.hr_height, opt_data.hr_width)

    save_path = os.path.join("./saved_models", "srgan", opt_total.save_path)
    model_path = os.path.join(save_path, "models")
    img_path = os.path.join(save_path, "images")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(opt_model.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    best_loss_G = 1e6
    best_loss_D = 1e6
    if opt_model.epoch < 0:
        best_epoch = 0
        for model_file in os.listdir("{}".format(model_path)):
            new_epoch = check_file_format(model_file)

            if new_epoch is None:
                continue

            best_epoch = max(best_epoch, new_epoch)

        opt_model.epoch = best_epoch

    if opt_model.epoch > 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(
            "{}/generator_{}.pth".format(model_path, opt_model.epoch),
            map_location=device
        ))
        discriminator.load_state_dict(torch.load(
            "{}/discriminator_{}.pth".format(model_path, opt_model.epoch),
            map_location=device
        ))

        with open(os.path.join(model_path, "best_loss.txt"), "r") as f:
            for line in f.readlines():
                if "Best generator loss:" in line:
                    # Extract the value after the colon and strip
                    # any whitespace or newline characters
                    best_loss_G = float(line.split(":")[1].strip())
                elif "Best discriminator loss:" in line:
                    # Extract the value after the colon and strip
                    # any whitespace or newline characters
                    best_loss_D = float(line.split(":")[1].strip())

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt_model.lr,
                                   betas=(opt_model.b1, opt_model.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt_model.lr,
                                   betas=(opt_model.b1, opt_model.b2))

    (train_loader, val_loader), (mean_torch, std_torch) = get_data(opt_data)

    # ----------
    #  Training
    # ----------

    for epoch in range(opt_model.epoch+1, opt_model.n_epochs+1):
        if epoch >= opt_model.epoch+3 and opt_total.test_mode:
            break

        train_loss_G = 0.0
        train_loss_D = 0.0
        train_loss_GAN = 0.0

        val_loss_G = 0.0
        val_loss_D = 0.0
        val_loss_GAN = 0.0

        train_hr_psnr = []
        train_lr_psnr = []
        train_ssim = []
        train_lpips = []

        val_hr_psnr = []
        val_lr_psnr = []
        val_ssim = []
        val_lpips = []

        for i, imgs in enumerate(train_loader):
            if i >= 2 and opt_total.test_mode:
                break

            batches_done = epoch * len(train_loader) + i

            loss_G, loss_D, loss_GAN, models_new, gen_hr = forward_pass(
                generator, discriminator, imgs,
                type_forward="train",
                optimizer_G=optimizer_G, optimizer_D=optimizer_D,
                loss_gan_rate=opt_model.loss_gan_rate
            )

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
            lpips_values = calculate_lpips_batch(
                imgs_hr, gen_hr, mean_std=(mean_torch, std_torch)
            )

            train_hr_psnr.append(hr_psnr_values)
            train_lr_psnr.append(lr_psnr_values)
            train_ssim.append(ssim_values)
            train_lpips.append(lpips_values)

            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()
            train_loss_GAN += loss_GAN.item()

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                "[GAN loss: %f] [G loss: %f]\n"
                % (epoch, opt_model.n_epochs, i, len(train_loader),
                   loss_D.item(), loss_GAN.item(), loss_G.item())
            )

            if batches_done % opt_model.sample_interval == 0:
                imgs_lr = imgs["lr"].to(device=device)
                # Save image grid with upsampled inputs and SRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_lr, gen_hr), -1)
                save_image(img_grid, "{}/{}.png".format(
                    img_path, batches_done
                ), normalize=False)

        train_loss_D /= len(train_loader)
        train_loss_G /= len(train_loader)
        train_loss_GAN /= len(train_loader)

        train_hr_psnr = np.mean(np.concatenate(train_hr_psnr))
        train_lr_psnr = np.mean(np.concatenate(train_lr_psnr))
        train_ssim = np.mean(np.concatenate(train_ssim))
        train_lpips = np.mean(np.concatenate(train_lpips))

        sys.stdout.write(
            "Total Training Stats: [Epoch %d/%d] [D loss: %f]"
            "[GAN loss: %f] [G loss: %f]\n"
            % (epoch, opt_model.n_epochs, train_loss_D,
               train_loss_GAN, train_loss_G)
        )
        sys.stdout.write(
            "[PSNR %f] [LR-PSNR %f] [SSIM %f] [LPIPS %f]\n"
            % (train_hr_psnr, train_lr_psnr, train_ssim, train_lpips)
        )

        with torch.no_grad():
            for i, imgs in enumerate(val_loader):
                if i >= 2 and opt_total.test_mode:
                    break

                loss_G, loss_D, loss_GAN, _, gen_hr = forward_pass(
                    generator, discriminator, imgs,
                    type_forward="eval",
                    loss_gan_rate=opt_model.loss_gan_rate
                )

                # Evaluate
                imgs_hr = imgs["hr"].to(device=device)

                hr_psnr_values = calculate_psnr_batch(
                    imgs_hr, gen_hr, mean_std=(mean_torch, std_torch))
                lr_psnr_values = calculate_psnr_batch(
                    imgs_hr, gen_hr, mode="lr",
                    mean_std=(mean_torch, std_torch)
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

        val_loss_D /= len(train_loader)
        val_loss_G /= len(train_loader)
        val_loss_GAN /= len(train_loader)

        val_hr_psnr = np.mean(np.concatenate(val_hr_psnr))
        val_lr_psnr = np.mean(np.concatenate(val_lr_psnr))
        val_ssim = np.mean(np.concatenate(val_ssim))
        val_lpips = np.mean(np.concatenate(val_lpips))

        sys.stdout.write(
            "Total Evaluation Stats: [Epoch %d/%d] [D loss: %f]"
            "[GAN loss: %f] [G loss: %f]\n"
            % (epoch, opt_model.n_epochs, val_loss_D, val_loss_GAN, val_loss_G)
        )
        sys.stdout.write(
            "[PSNR %f] [LR-PSNR %f] [SSIM %f] [LPIPS %f]\n"
            % (val_hr_psnr, val_lr_psnr, val_ssim, val_lpips)
        )

        if (
            opt_model.checkpoint_interval != -1
            and epoch % opt_model.checkpoint_interval == 0
        ):
            # Save model checkpoints
            torch.save(generator.state_dict(),
                       "{}/generator_{}.pth".format(model_path, epoch))
            torch.save(discriminator.state_dict(),
                       "{}/discriminator_{}.pth".format(model_path, epoch))

        if val_loss_G < best_loss_G and val_loss_D < best_loss_D:
            # Save model checkpoints
            torch.save(generator.state_dict(),
                       "{}/generator_best.pth".format(model_path))
            torch.save(discriminator.state_dict(),
                       "{}/discriminator_best.pth".format(model_path))

            with open(os.path.join(model_path, "best_loss.txt"), "w") as f:
                f.write(f"Best generator loss: {val_loss_G}\n")
                f.write(f"Best discriminator loss: {val_loss_D}\n")
                f.write(f"Best epoch: {epoch}")

            best_loss_G = val_loss_G
            best_loss_D = val_loss_D

            sys.stdout.write(
                "Save best checkpoint: Epoch %d/%d\n"
                % (epoch, opt_model.n_epochs)
            )
        sys.stdout.write(
            "--------------Finish Epoch %d/%d-----------------------------\n"
            % (epoch, opt_model.n_epochs)
        )


class Runner:
    def __init__(self):
        pass
