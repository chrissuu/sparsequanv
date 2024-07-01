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

def relu(x):
    _relu = nn.ReLU()
    
    return _relu(x)
# change to global min / max
def curly_N(w):
    w_min, w_max = torch.min(torch.min(torch.min(w))), torch.max(torch.max(torch.max(w)))
    reg_N = (w - w_min) / (w_max - w_min)
    # print(reg_N.min(), reg_N.max())
    return reg_N

def curly_Nprime(w):
    # print(type(w))
    w_min, w_max = torch.min(torch.min(torch.min(w))), torch.max(torch.max(torch.max(w)))
    curly_N = (w - w_min + 1) / (w_max - w_min + 2)
    return curly_N
    # return (w - torch.min(w) + 1) / (torch.max(w) - torch.min(w) + 2)

def f_VHN(x, w):
    relu_x = relu(curly_N(x))
    relu_w = relu(curly_Nprime(w))
    
    return relu_x * relu_w

def min_max(x):
    
    return curly_Nprime(x)

class VHNLayer(nn.Module):
    """ Custom VHN layer """
    def __init__(self, bz, img_height, img_len, img_width):
        super().__init__()
        self.bz, self.img_height, self.img_len, self.img_width = bz, img_height, img_len, img_width
        weights = torch.ones(size = (img_height, img_len, img_width)).float()
        weights = min_max(weights)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        
        # initialize weights and biases
        

    def forward(self, x):
        res = torch.tensor(np.zeros((self.bz, self.img_height, self.img_len, self.img_width)))
        
        res = f_VHN(x, self.weights)

        return res.type(torch.float32)
    
