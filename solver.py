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
from itertools import count
from torchvision import utils

class Solver():
    def __init__(self, parser, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.mode = parser.mode
        self.input_size = parser.input_size
        self.batch_size = parser.batch_size
        self.epoch = parser.epoch
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
        
        for epoch in range(1, self.epoch+1):
            data1 = iter(self.loader1)
            data2 = iter(self.loader2)
            while True:
                try:
                    batch1, _ = data1.next()
                    batch1 = self.__make_var(batch1)
                    batch2, _ = data2.next()
                    batch2 = self.__make_var(batch2)
                    
                    # Train discriminator
                    
                    # Real images
                    self.__reset_grad()
                    
                    out1 = self.d1(batch1)
                    d1_loss = torch.mean((out1-1)**2)
                    
                    out2 = self.d1(batch2)
                    d2_loss = torch.mean((out2-1)**2)
                    
                    d_real_loss = d1_loss + d2_loss
                    d_real_loss.backward()
                    self.d_optimizer.step()
                    
                    # Fake images
                    self.__reset_grad()
                    
                    fake_batch1 = self.g21(batch2) # G21(image2)
                    out2 = self.d1(fake_batch1) # D1(G21(image2))
                    d1_loss = torch.mean(out2**2)
                    
                    fake_batch2 = self.g12(batch1) # G12(image1)
                    out1 = self.d2(fake_batch2) # D2(G12(image1))
                    d2_loss = torch.mean(out1**2)
                    
                    
                    d_fake_loss = d1_loss + d2_loss
                    d_fake_loss.backward()
                    self.d_optimizer.step()
                    
                    # Discriminator trained
                    
                    # Train generators
                    
                    # Generator12
                    self.__reset_grad()
                    
                    fake_batch2 = self.g12(batch1) #G12(batch1)
                    out2 = self.d2(fake_batch2)
                    reconst_batch1 = self.g21(fake_batch2) #G21(G12(batch1))
                    
                    g12_loss = torch.mean((out2-1)**2)
                    g12_loss += torch.mean((batch1 - reconst_batch1)**2) #reconst loss
                    g12_loss.backward()
                    self.g_optimizer.step()
                    
                    # Generator21
                    self.__reset_grad()
                    fake_batch1 = self.g21(batch2) #G21(batch2)
                    out1 = self.d1(fake_batch1)
                    reconst_batch2 = self.g12(fake_batch1) #G12(G21(batch2))
                    
                    g21_loss = torch.mean((out1-1)**2)
                    g21_loss += torch.mean((batch2 - reconst_batch2)**2)
                    g21_loss.backward()
                    self.g_optimizer.step()
                    
                    # Generators trained
                            
                except StopIteration:
                    logging.info("Epoch [%d/%d] d_real_loss=%.5f d_fake_loss=%.5f g12_loss=%.5f g21_loss=%.5f"
                    %(epoch, self.epoch, d_real_loss, d_fake_loss, g12_loss, g21_loss))
                    break
        logging.info("Training done")
        data1 = iter(self.loader1)
        data2 = iter(self.loader2)
        batch1, _ = data1.next()
        batch2, _ = data2.next()
        batch1 = self.__make_var(batch1)
        batch2 = self.__make_var(batch2)
        tensor2 = self.g12(batch1)
        tensor1 = self.g21(batch2)
        utils.save_image(tensor2[:, :, :], "tensor2.png")
        utils.save_image(tensor1[:, :, :], "tensor1.png")
        return 1
