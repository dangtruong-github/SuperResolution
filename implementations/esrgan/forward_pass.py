import torch
from torch.autograd import Variable

import numpy as np

from implementations.esrgan.models import FeatureExtractor


cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
allowed_types = ["train", "eval"]

feature_extractor = FeatureExtractor().to(device)

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

def forward_pass(
    generator, discriminator, imgs,
    type_forward="train",
    optimizer_G=None, optimizer_D=None,
    loss_gan_rate=5e-3, loss_pixel_rate=1e-2,
    warm_up_batch=False
):
    if type_forward not in allowed_types:
        raise ValueError(f"{type_forward} not allowed. Try {allowed_types}")

    if type_forward == "train":
        generator.train()
        discriminator.train()
    else:
        generator.eval()
        discriminator.eval()
    
    # Configure model input
    imgs_lr = Variable(imgs["lr"].type(Tensor))
    imgs_hr = Variable(imgs["hr"].type(Tensor))

    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

    # ------------------
    #  Train Generators
    # ------------------
    if type_forward == "train":
        optimizer_G.zero_grad()

    # Generate a high resolution image from low resolution input
    gen_hr = generator(imgs_lr)

    # Measure pixel-wise loss against ground truth
    loss_pixel = criterion_pixel(gen_hr, imgs_hr)

    if warm_up_batch:
        if type_forward == "eval":
            raise ValueError("Warm-up batch cannot be evaluate")
        # Warm-up (pixel-wise loss only)
        loss_pixel.backward()
        optimizer_G.step()

        return loss_pixel, (generator, discriminator, optimizer_G, optimizer_D)
        

    # Extract validity predictions from discriminator
    pred_real = discriminator(imgs_hr).detach()
    pred_fake = discriminator(gen_hr)

    # Adversarial loss (relativistic average GAN)
    loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

    # Content loss
    gen_features = feature_extractor(gen_hr)
    real_features = feature_extractor(imgs_hr).detach()
    loss_content = criterion_content(gen_features, real_features)

    # Total generator loss
    loss_G = loss_content + loss_gan_rate * loss_GAN + loss_pixel_rate * loss_pixel

    if type_forward == "train":
        loss_G.backward()
        optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    if type_forward == "train":
        optimizer_D.zero_grad()

    pred_real = discriminator(imgs_hr)
    pred_fake = discriminator(gen_hr.detach())

    # Adversarial loss for real and fake images (relativistic average GAN)
    loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
    loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

    # Total loss
    loss_D = (loss_real + loss_fake) / 2

    if type_forward == "train":
        loss_D.backward()
        optimizer_D.step()

    return loss_G, loss_D, (loss_GAN, loss_content, loss_pixel), (generator, discriminator, optimizer_G, optimizer_D), gen_hr
