"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import sys

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from inference import EvaluateLoader

import torch.nn as nn
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

import sys

# Open the file and redirect sys.stdout
sys.stdout = open('output.txt', 'w')  # Redirect stdout to file

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
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

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_{}.pth".format(opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_{}.pth".format(opt.epoch)))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s/train" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_loader = DataLoader(
    ImageDataset("../../data/%s/val" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    train_loss_G = 0.0
    train_loss_D = 0.0

    if epoch == opt.epoch:
        pass
        """
        val_loss, val_metrics = EvaluateLoader(
            generator, discriminator, feature_extractor, val_loader,
            mean_std=(dataloader.dataset.mean, dataloader.dataset.std)
        )
        print(val_loss)
        print(val_metrics)
        """

    # Initialize the WandbLogger
    #wandb_logger = WandbLogger(project='your-project-name')

    # Set up the Trainer with the logger
    #trainer = Trainer(logger=wandb_logger)

    generator.train()
    discriminator.train()
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

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

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
        # --------------
        #  Log Progress
        # --------------

        train_loss_G += loss_G.item()
        train_loss_D += loss_D.item()

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    val_loss, val_metrics = EvaluateLoader(
        generator, discriminator, feature_extractor, val_loader,
        mean_std=(val_loader.dataset.mean, val_loader.dataset.std)
    )

    val_loss_G, val_loss_D = val_loss
    val_psnr, val_ssim, val_lpips, val_lr_psnr = val_metrics

    avg_train_loss_G = train_loss_G / len(dataloader)
    avg_train_loss_D = train_loss_D / len(dataloader)

    print(f"Epoch: {epoch}/{opt.n_epochs}:")
    print(f"Train generative loss: {avg_train_loss_G}")
    print(f"Train discriminative loss: {avg_train_loss_D}")
    print(f"Validation generative loss: {val_loss_G}")
    print(f"Validation discriminative loss: {val_loss_D}")
    print(f"Validation PSNR score: {val_psnr}")
    print(f"Validation SSIM score: {val_ssim}")
    print(f"Validation LPIPS score: {val_lpips}")
    print(f"Validation LR-PSNR score: {val_lr_psnr}")

    with open("./train.log", "a") as f:
        f.write(f"Epoch: {epoch}/{opt.n_epochs}:\n")
        f.write(f"Train generative loss: {avg_train_loss_G}\n")
        f.write(f"Train discriminative loss: {avg_train_loss_D}\n")
        f.write(f"Validation generative loss: {val_loss_G}\n")
        f.write(f"Validation discriminative loss: {val_loss_D}\n")
        f.write(f"Validation PSNR score: {val_psnr}\n")
        f.write(f"Validation SSIM score: {val_ssim}\n")
        f.write(f"Validation LPIPS score: {val_lpips}\n")
        f.write(f"Validation LR-PSNR score: {val_lr_psnr}\n")

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)


# Print statements that will be written to the file
print("This text will be written to the file.")
print("Another line in the file.")

# Close the file and restore sys.stdout to its original state
sys.stdout.close()