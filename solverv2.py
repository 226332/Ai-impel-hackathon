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
from torch import optim, save
from modelv2 import Generator256, Discriminator256
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
        self.g12.normal_weight_init()
        self.g21.normal_weight_init()
        self.d1.normal_weight_init()
        self.d2.normal_weight_init()
        device = torch.device("cuda:0")
        self.g12 = nn.DataParallel(self.g12, device_ids=[0,1,2,3])
        self.g21 = nn.DataParallel(self.g21, device_ids=[0,1,2,3])
        self.d1 = nn.DataParallel(self.d1, device_ids=[0,1,2,3])
        self.d2 = nn.DataParallel(self.d2, device_ids=[0,1,2,3])
        self.g12.to(device)
        self.g21.to(device)
        self.d1.to(device)
        self.d2.to(device)

        self.g_optimizer = self.__get_generator_optimizer()
        self.d1_optimizer = optim.Adam(self.d1.parameters(), self.lr, [0.5, 0.999])
        self.d2_optimizer = optim.Adam(self.d2.parameters(), self.lr, [0.5, 0.999])

    def __get_generators(self):
        return Generator256(), Generator256()

    def __get_discriminators(self):
        return Discriminator256(), Discriminator256()

    def __get_generator_optimizer(self):
        return optim.Adam(
            list(self.g12.parameters()) + list(self.g21.parameters()), self.lr,
            [0.5, 0.999])

    def __make_var(self, tensor):
        tensor = tensor.cuda()
        return Variable(tensor)

    def train(self):
        logging.info("Train started")

        MSE_loss = torch.nn.MSELoss().cuda()
        L1_loss = torch.nn.L1Loss().cuda()

        for epoch in range(1, self.epoch + 1):
            data1 = iter(self.loader1)
            data2 = iter(self.loader2)
            while True:
                try:
                    batch1, _ = data1.next()
                    batch1 = self.__make_var(batch1)
                    batch2, _ = data2.next()
                    batch2 = self.__make_var(batch2)

                    # Train generators

                    # Generator12
                    fake_batch2 = self.g12(batch1)
                    fake_2_decision = self.d2(fake_batch2)
                    g12_loss = MSE_loss(fake_2_decision, Variable(torch.ones(fake_2_decision.size()).cuda()))

                    reconst_batch1 = self.g21(fake_batch2)
                    reconst_g12_loss = L1_loss(reconst_batch1, batch1) * 10

                    # Generator21
                    fake_batch1 = self.g21(batch2)
                    fake_1_decision = self.d1(fake_batch1)
                    g21_loss = MSE_loss(fake_1_decision, Variable(torch.ones(fake_1_decision.size()).cuda()))

                    reconst_batch2 = self.g12(fake_batch1)
                    reconst_g21_loss = L1_loss(reconst_batch2 , batch2) * 10

                    G_loss = g12_loss + g21_loss + reconst_g12_loss + reconst_g21_loss
                    self.g_optimizer.zero_grad()
                    G_loss.backward()
                    self.g_optimizer.step()

                    # Train discriminators 1
                    D_1_real_decision = self.d1(batch1)
                    D_1_real_loss = MSE_loss(D_1_real_decision, Variable(torch.ones(D_1_real_decision.size()).cuda()))
                    fake_batch1 = self.g21(batch2)
                    D_1_fake_decision = self.d1(fake_batch1)
                    D_1_fake_loss  = MSE_loss(D_1_fake_decision, Variable(torch.zeros(D_1_fake_decision.size()).cuda()))

                    D_1_loss = (D_1_real_loss + D_1_fake_loss) * 0.5
                    self.d1.zero_grad()
                    D_1_loss.backward()
                    self.d1_optimizer.step()

                    # Train discriminators 2
                    D_2_real_decision = self.d2(batch2)
                    D_2_real_loss = MSE_loss(D_2_real_decision, Variable(torch.ones(D_2_real_decision.size()).cuda()))
                    fake_batch2 = self.g12(batch1)
                    D_2_fake_decision = self.d2(fake_batch2)
                    D_2_fake_loss = MSE_loss(D_2_fake_decision, Variable(torch.zeros(D_2_fake_decision.size()).cuda()))

                    D_2_loss = (D_2_real_loss + D_2_fake_loss) * 0.5
                    self.d2.zero_grad()
                    D_2_loss.backward()
                    self.d2_optimizer.step()

                except StopIteration:
                    logging.info(
                        "Epoch [%d/%d] d1_loss=%.5f d2_loss=%.5f g_loss=%.5f"
                        % (epoch, self.epoch, D_1_loss.data, D_2_loss.data, G_loss.data))
                    self.save_output_tensors()
                    break
        logging.info("Training done")
        self.save_output_tensors()
        self.save_params()
        return 1

    def save_output_tensors(self):
        data1 = iter(self.loader1).next()[0]
        data2 = iter(self.loader2).next()[0]
        utils.save_image(data1[:, :, :], "./samples/B_output_before.png")
        utils.save_image(data2[:, :, :], "./samples/A_output_before.png")
        batch1 = self.__make_var(data1)
        batch2 = self.__make_var(data2)
        tensor2 = self.g12(batch1)
        tensor1 = self.g21(batch2)
        utils.save_image(tensor2[:, :, :], "./samples/B_output.png")
        utils.save_image(tensor1[:, :, :], "./samples/A_output.png")
    
    def save_params(self):
        save(self.g12.state_dict(), "./models_params/g12.pkl")
        save(self.g21.state_dict(), "./models_params/g21.pkl")
        save(self.d1.state_dict(), "./models_params/g1.pkl")
        save(self.d2.state_dict(), "./models_params/d2.pkl")
