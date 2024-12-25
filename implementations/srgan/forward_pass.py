import torch
from torch.autograd import Variable

import numpy as np

from implementations.srgan.models import FeatureExtractor


cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

feature_extractor = FeatureExtractor().to(device)

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

allowed_types = ["train", "eval"]

def forward_pass(
    generator, discriminator, imgs, 
    type_forward="train",
    optimizer_G=None, optimizer_D=None,
    loss_gan_rate=1e-3
):
    # Configure model input
    if type_forward not in allowed_types:
        raise ValueError(f"{type_forward} not allowed. Try {allowed_types}")

    if type_forward == "train":
        generator.train()
        discriminator.train()
    else:
        generator.eval()
        discriminator.eval()

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

    # Adversarial loss
    loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

    # Content loss
    gen_features = feature_extractor(gen_hr)
    real_features = feature_extractor(imgs_hr)
    loss_content = criterion_content(gen_features, real_features.detach())

    # Total loss
    loss_G = loss_content + loss_gan_rate * loss_GAN
    
    if type_forward == "train":
        loss_G.backward()
        optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    if type_forward == "train":
        optimizer_D.zero_grad()

    # Loss of real and fake images
    loss_real = criterion_GAN(discriminator(imgs_hr), valid)
    loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

    # Total loss
    loss_D = (loss_real + loss_fake) / 2

    if type_forward == "train":
        loss_D.backward()
        optimizer_D.step()

    # --------------
    #  Log Progress
    # --------------

    return loss_G, loss_D, loss_GAN, (generator, discriminator, optimizer_G, optimizer_D), gen_hr