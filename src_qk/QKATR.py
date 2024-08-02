import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auprc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from SparseQuanvLayer import SparseQuanvLayer

class ATR(nn.Module):
    def __init__(self, nc, bz, device = None, N_filters=4, N_output = 1, ker = 4, s = 2, pad =1):
        super(ATR, self).__init__()
        self.ker = ker
        self.s = s
        self.pad = pad
        self.nc = nc
        self.bz = bz
        # print(nc)
        # self.device = device
        self.N_filters = N_filters
        self.N_output = N_output
        self.conv1 = nn.Conv3d(nc, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv2 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv3 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        
        self.avgpool = nn.AvgPool3d(kernel_size = (6, 2, 2), stride= (1, 1, 1), padding= (0, 0, 0))
    

        # "columns input x and output columnes"

        self.f2 = nn.Linear(1536,1)
        

        self.sigmoid = nn.Sigmoid()


    def forward(self, x0):
        "image vectorization"
        # print(x0.shape)
        # x0.unsqueeze(1)
        # print(x0.shape)
        # print(self.nc)
        x0 = x0.reshape((self.bz, self.nc, 32, 32, 50)).float()
        x = self.conv1(x0)
        x = self.avgpool(F.relu(x))
        x = self.conv2(x)
        x = self.avgpool(F.relu(x))
        x = self.conv3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.f2(x)
        # print(x)
        y = self.sigmoid(x)
        # print(y)

        return y
        

class QKATR(nn.Module):
    def __init__(self, nc, bz, rand_params, wires, q_dev, N_filters=4, PRINT = True):
        super(ATR, self).__init__()
        self.nc = nc
        self.N_filters = N_filters
        self.bz = bz
        self.print = PRINT
        # self.device = device
        self.N_filters = N_filters
        self.sparsequanv = SparseQuanvLayer(rand_params, wires, q_dev, input_shape = (64,64,101), kernel_shape = (2,2,2), bz= bz, PRINT=PRINT)
        self.conv1 = nn.Conv3d(nc, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv2 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv3 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        
        self.avgpool = nn.AvgPool3d(kernel_size = (6, 2, 2), stride= (1, 1, 1), padding= (0, 0, 0))
    

        # "columns input x and output columnes"

        self.f2 = nn.Linear(1536,1)
        

        self.sigmoid = nn.Sigmoid()


    def forward(self, x0):
        "image vectorization"
        # print(x0.shape)
        # x0.unsqueeze(1)
        # print(x0.shape)
        x0 = x0.reshape((self.bz, self.nc, 32, 32, 50)).float()
        x = self.conv1(x0)
        x = self.avgpool(F.relu(x))
        x = self.conv2(x)
        x = self.avgpool(F.relu(x))
        x = self.conv3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.f2(x)
        # print(x)
        y = self.sigmoid(x)
        # print(y)

        return y
            
        
