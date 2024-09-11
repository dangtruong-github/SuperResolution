import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2

import wandb
import sys

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

def train(args):
    print(torch.cuda.is_available())
    logfile = open("train.log", "w")

    # Redirect stdout to both console and the file
    sys.stdout = Tee(sys.stdout, logfile)
    """wandb.init(
        # Set the project where this run will be logged
        project="super-resolution",
        # Track hyperparameters and run metadata
        # config=args
    )"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform  = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = args.in_memory, transform = transform)
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num, scale=args.scale)
    
    if args.fine_tuning:        
        generator.load_state_dict(torch.load(args.generator_path, map_location=device), strict=False)
        print("pre-trained model is loaded haha")
        print("path : %s"%(args.generator_path))
        
    generator = generator.to(device)
    generator.train()
    
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr = 1e-4)
        
    pre_epoch = 0
    fine_epoch = 0

    print("Start pre-training process")
    
    #### Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
        print(f"Epoch: {pre_epoch}/{args.pre_train_epoch}:")

        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{len(loader)}:", loss.item())

        # wandb.log({"loss": loss})
        pre_epoch += 1

        if pre_epoch % 1 == 0:
            print(pre_epoch)
            print(loss.item())
            print('=========')

        if pre_epoch % 800 ==0:
            torch.save(generator.state_dict(), './model/pre_trained_model_%03d.pt'%pre_epoch)

        
    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()
    
    discriminator = Discriminator(patch_size = args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()
    
    d_optim = optim.Adam(discriminator.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size = 2000, gamma = 0.1)
    
    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    
    print("Start fine-tuning process")

    while fine_epoch < args.fine_train_epoch:
        
        scheduler.step()
        print(f"Epoch: {fine_epoch}/{args.fine_train_epoch}:")
        
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            real_label = torch.ones((gt.shape[0], 1)).to(device)
            fake_label = torch.zeros((gt.shape[0], 1)).to(device)
                        
            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer = args.feat_layer)
            
            L2_loss = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat)**2)
            
            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss
            
            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{len(loader)}:\n")
                print("Generative loss:", g_loss.item())
                print("Discriminative loss:", d_loss.item())

            
        fine_epoch += 1

        if fine_epoch % 1 == 0:
            print(f"Epoch: {fine_epoch}/{args.fine_train_epoch}:")
            print(g_loss.item())
            print(d_loss.item())
            print('=========')

        if fine_epoch % 500 ==0:
            #torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            #torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)
            torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)

    logfile.close()


def test(args):
    print(torch.cuda.is_available())
    """wandb.init(
        # Set the project where this run will be logged
        project="super-resolution",
        # Track hyperparameters and run metadata
        config=args
    )"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num).to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=torch.device(device)))
    generator.eval()
    
    f = open('./result.txt', 'w')
    psnr_list = []
    psnr_lr_list = []
    ssim_list = []
    lpips_list = []

    # Load the LPIPS model
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w * args.scale]

            #print(next(generator.parameters()).is_cuda)
            #print(lr.is_cuda)

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            # Normalize images to [0,1]
            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)

            # Convert to Y channel
            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

            # PSNR for high-resolution output
            psnr_hr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr_hr)
            
            # PSNR for low-resolution input (comparing lr to downsampled gt)
            lr_resized = cv2.resize(gt, (w, h), interpolation=cv2.INTER_LINEAR)  # Downsample gt to match lr size
            y_lr_resized = rgb2ycbcr(lr_resized)[:, :, :1]
            y_lr = rgb2ycbcr(lr[0].cpu().numpy().transpose(1, 2, 0))[:, :, :1]
            psnr_lr = peak_signal_noise_ratio(y_lr / 255.0, y_lr_resized / 255.0, data_range=1.0)
            psnr_lr_list.append(psnr_lr)
            
            # SSIM metric
            #print(y_output.shape)
            #print(y_gt.shape)
            ssim_val = ssim(np.squeeze(y_output) / 255.0, np.squeeze(y_gt) / 255.0, data_range=1.0, multichannel=True)
            ssim_list.append(ssim_val)

            # LPIPS metric
            lpips_val = loss_fn(torch.from_numpy(output.transpose(2, 0, 1)).unsqueeze(0).to(device), 
                                torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).to(device))
            lpips_list.append(lpips_val.item())

            # Save the output image locally
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png' % i)

            # Log images and metrics to wandb
            """wandb.log({
                "PSNR_HR": psnr_hr,
                "PSNR_LR": psnr_lr,
                "SSIM": ssim_val,
                "LPIPS": lpips_val.item(),
                "Output Image": [wandb.Image(result, caption=f"Output Image {i}")],
                "Ground Truth": [wandb.Image(Image.fromarray((gt * 255.0).astype(np.uint8)), caption=f"Ground Truth {i}")]
            })"""
            f.write(f'psnr batch : {psnr_hr}')
            f.write(f'psnr_lr batch : {psnr_lr}')
            f.write(f'ssim batch : {ssim_val}')
            f.write(f'lpips batch : {lpips_val}')

        # Optionally, store all metrics at the end of the loop in wandb tables/logs
        metrics = {'PSNR_HR': psnr_list, 'PSNR_LR': psnr_lr_list, 'SSIM': ssim_list, 'LPIPS': lpips_list}
        #wandb.log(metrics)
        f.write('avg psnr : %04f' % np.mean(psnr_list))
        f.write('avg ssim : %04f' % np.mean(ssim_list))
        f.write('avg psnr_lr : %04f' % np.mean(psnr_lr_list))
        f.write('avg lpips : %04f' % np.mean(lpips_list))


def test_only(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            lr = te_data['LR'].to(device)
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1,2,0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/res_%04d.png'%i)



