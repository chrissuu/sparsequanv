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

from Datacompiler import generate_compiler
from utils import *
from train_test import test
from train_test import train
from SparsifiedDataset import QuantumDataset
from SparseQuanvLayer import SparseQuanvLayer

class SQNN(nn.Module):
    def __init__(self, bz, rand_params, w,  q_dev, PRINT=True, device = None, N_filters=4, N_output = 1, ker = 4, s = 2, pad =1):
        super(SQNN, self).__init__()
        self.ker = ker
        self.s = s
        self.pad = pad
        self.bz = bz
        self.WIRES = w
        self.q_dev = q_dev
        # self.device = device
        self.N_filters = N_filters
        self.N_output = N_output
        self.rand_params = rand_params
        self.sparsequanv = SparseQuanvLayer(rand_params, w, q_dev, input_shape = (64,64,101), kernel_shape = (2,2,2), bz= bz, PRINT=PRINT)
        self.conv1 = nn.Conv3d(w, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv2 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        
        self.avgpool = nn.AvgPool3d(kernel_size = (6, 2, 2), stride= (1, 1, 1), padding= (0, 0, 0))
    

        # "columns input x and output columnes"

        self.f2 = nn.Linear(22000,8832)
        self.f3 = nn.Linear(8832,2148)
        self.f4 = nn.Linear(2148,1)

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x0):
        "image vectorization"
        # print(x0.shape)
        # x0.unsqueeze(1)
        print(type(x0))
        
        x = self.sparsequanv.forward(x0)
        print(x.shape)
        x = x.reshape((self.bz, self.WIRES, 32, 32, 50))
        x = self.conv1(x0)
        x = self.avgpool(F.relu(x))
        x = self.conv2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        y = self.sigmoid(x)

        return y