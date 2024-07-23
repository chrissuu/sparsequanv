import os
import time
import h5py
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torcheval
from torcheval.metrics.functional import binary_auprc
import matplotlib.pyplot as plt
from utils import VHNLayer


class ATR(nn.Module):
    def __init__(self, nc, bz, wires, device = None, N_filters=4, N_output = 1, ker = 4, s = 2, pad =1):
        super(ATR, self).__init__()
        self.ker = ker
        self.s = s
        self.pad = pad
        self.nc = nc
        self.bz = bz
        self.wires = wires
        # self.device = device
        self.N_filters = N_filters
        self.N_output = N_output
        self.conv1 = nn.Conv3d(nc, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv2 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv3 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv4 = nn.Conv3d(N_filters, 1, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.avgpool = nn.AvgPool3d(kernel_size = (6, 2, 2), stride= (1, 1, 1), padding= (0, 0, 0))
    

        # "columns input x and output columnes"

        self.f2 = nn.Linear(1248,1)

        self.sigmoid = nn.Sigmoid()


        

    def forward(self, x0):
        "image vectorization"
        # print(x0.shape)
        # print(x0.shape)
        # print(type(x0))
        x0 = (x0.reshape((self.bz, self.wires, 101, 64, 64))).float()
        x = self.conv1(x0)
        x = self.avgpool(F.relu(x))
        x = self.conv2(x)
        x = self.avgpool(F.relu(x))
        x = self.conv3(x)
        x = self.avgpool(F.relu(x))
        x = self.conv4(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.f2(x)

        y = self.sigmoid(x)

        return y

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
 
