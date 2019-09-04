"""*****************************************************************************
Solver is able to train and test images

*****************************************************************************"""
import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.misc
import numpy as np
from torch.autograd import Variable
from torch import optim
from model import Generator, Discriminator
import logging

class Solver():
    def __init__(self, parser, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.mode = parser.mode
        self.input_size = parser.input_size
        self.batch_size = parser.batch_size
        self.train_iter = parser.train_iter
        self.device = torch.device(f"cuda:{parser.gpu}")
        self.conv_dim = parser.conv_dim
        self.lr = parser.lr
        self.g12, self.g21 = self.__get_generators()
        self.d1, self.d2 = self.__get_discriminators()
        self.g_optimizer = self.__get_generator_optimizer()
        self.d_optimizer = self.__get_discriminator_optimizer()
        self.g12.cuda()
        self.g21.cuda()
        self.d1.cuda()
        self.d2.cuda()

    def __get_generators(self):
        return Generator(conv_dim=self.conv_dim), Generator(conv_dim=self.conv_dim)

    def __get_discriminators(self):
        return Discriminator(conv_dim=self.conv_dim), Discriminator(conv_dim=self.conv_dim)

    def __get_generator_optimizer(self):
        return optim.Adam(list(self.g12.parameters()) + list(self.g21.parameters()), self.lr, [0.5, 0.999])

    def __get_discriminator_optimizer(self):
        return optim.Adam(list(self.d1.parameters()) + list(self.d2.parameters()), self.lr, [0.5, 0.999])

    def __make_var(self, tensor):
        tensor = tensor.cuda()
        return Variable(tensor)

    def __make_cpu(self, tensor):
        tensor = tensor.cpu()
        return tensor.data.numpy()

    def __reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        logging.info("Train started")
        return 1
