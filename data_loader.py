import os

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


def get_transforms_list(opts):
    transforms_list = []

    return transforms_list


def get_data_loader(opts):
    transforms_list = get_transforms_list(opts)

    transform = transforms.Compose([
        transforms.Scale(opts.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_path = os.path.join('./data', opts.dataset_name)
    test_path = os.path.join('./data', f'Test_{opts.dataset_name}')

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset,
                               batch_size=opts.batch_size, shuffle=True,
                               num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size,
                              shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader



