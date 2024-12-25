import torch
from torch.utils.data import DataLoader

import numpy as np

from datasets.loader import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(opt):
    mean_np = np.load("data/{}/mean.npy".format(opt.train_set))
    std_np = np.load("data/{}/std.npy".format(opt.train_set))

    mean_torch = torch.from_numpy(mean_np)
    std_torch = torch.from_numpy(std_np)

    train_loader = DataLoader(
        ImageDataset(
            "data/%s/hr" % opt.train_set, 
            mean=mean_torch, std=std_torch
        ), #s, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu*2,
    )

    val_loader = DataLoader(
        ImageDataset(
            "data/%s/hr" % opt.val_set, 
            mean=mean_torch, std=std_torch
        ), #s, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu*2,
    )

    mean_torch = mean_torch.to(device=device)
    std_torch = std_torch.to(device=device)

    return (train_loader, val_loader), (mean_torch, std_torch)
