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
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import binary_auprc
from utils import *
import time
import sklearn.metrics

def relu(x):
    _relu = nn.ReLU()
    
    return _relu(x)
# change to global min / max
def curly_N(w):
    w_min, w_max = torch.min(torch.min(torch.min(w))), torch.max(torch.max(torch.max(w)))
    reg_N = (w - w_min) / (w_max - w_min)
    print(reg_N.min(), reg_N.max())
    return reg_N

def curly_Nprime(w):
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
    def __init__(self, channels, img_len, img_width):
        super().__init__()
        self.channels, self.img_len, self.img_width = channels, img_len, img_width
        weights = torch.Tensor(channels, img_len, img_width)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        
        

    def forward(self, x):
        
        return f_VHN(x, self.weights) 
    

def train(criterion1, criterion2, optimizer, net, num_epochs, dldr_trn):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dldr_trn, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            # print(f"Inputs shape: {inputs.shape}")
            
            temp_inputs = (torch.log10(inputs + 1)).float().squeeze(0)
            # print(f"Inputs shape post: {inputs.shape}")

            # print(inputs.shape)
            temp_labels = labels
            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs)
            # forward + backward + optimize
            # print(f"TEMP INPUTS TYPE: {type(temp_inputs.item())}")
            outputs = net(temp_inputs)
            loss = criterion1(outputs, temp_labels.reshape(dldr_trn.batch_sampler.batch_size,1).type(torch.float32))
            if criterion2: 
                # print("SHAPES")
                # print(curly_Nprime(net.vhn.weights).shape)
                # print(torch.sum(temp_inputs, dim = 0).shape)
                # print(temp_inputs.shape)
                _temp = temp_inputs.reshape((dldr_trn.batch_sampler.batch_size, net.WIRES, 32, 32, 50))
                x_bar = np.zeros((dldr_trn.batch_sampler.batch_size, net.WIRES, 32, 32, 50), dtype='float')
                target_cnt = 0
                for i in range(dldr_trn.batch_sampler.batch_size):
                    if int(temp_labels[i]) == 1:
                        x_bar = np.add(x_bar, _temp[i])
                        target_cnt += 1
                x_bar /= target_cnt
                loss += criterion2(curly_Nprime(net.vhn.weights), curly_N(x_bar.float()))
                loss = loss.float()
                
            # print("netvhn", net.vhn.weights.shape)
            # print(curly_N(torch.sum(inputs, dim = 0) / dldr.batch_size).shape)
       
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            # if i % 5 == 4:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
              
            #     running_loss = 0.0

    return

def test(net, dldr_tst):
    preds = []
    labels = []
    # imax = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dldr_tst,0):
            inputs, label = data
            
            temp_inputs = torch.log10(inputs + 1).float().squeeze(0)
            # print(inputs.shape)
            temp_label = label
            # calculate outputs by running images through the 
            output = net(temp_inputs)

            preds.append(output.tolist())
            labels.append(temp_label.tolist())
            # imax.append(i)
            # print(inputs.shape)
    preds_parsed = []
    labels_parsed = []
    # print(f"num_test {i}".format(i = max(imax)))
    for i, pred_list in enumerate(preds):
        total += len(pred_list)
        for k, pred in enumerate(pred_list):
            preds_parsed.append(pred[0])
            # if pred[0] >= 0.5:
            #     preds_parsed.append([1])
            # else:
            #     preds_parsed.append([0])
            labels_parsed.append(labels[i][k])
            if pred[0] >= 0.5: 
                if labels[i][k] == 1:
                    correct += 1
            else:
                if labels[i][k] == 0:
                    correct += 1

    
    # print(preds_parsed)
    # print(labels_parsed)
    # print()
    # print(labels_parsed)
    # print(preds_parsed)
    # print(preds_parsed)
    # print(labels_parsed)
    aucpr = sklearn.metrics.average_precision_score(labels_parsed, preds_parsed)

    return correct/total, aucpr, f"ACCURACY {correct / total}", f"PRAUC {aucpr}"


# configs = (criterion, optimizer, NUM_EPOCHS, TEST_SKIPS)
# data = (dldr_trn, dldr_tst)
def tt_print(net, data, configs):
    (criterion1, optimizer, num_epochs, skip) = (configs)
    (dldr_trn, dldr_tst) = (data)

    arr_epoch = [i for i in range(0, num_epochs)]
    aucpr_tst = []
    losses = []
    inference_times = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch %5 == 0:
            print(f"starting epoch {epoch}")
        running_loss = 0.0
        for i, data in enumerate(dldr_trn, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            # print(f"Inputs shape: {inputs.shape}")
            
            temp_inputs = (torch.log10(inputs + 1)).float().squeeze(0)
            # print(f"Inputs shape post: {inputs.shape}")

            # print(inputs.shape)
            temp_labels = labels
            # print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs)
            # forward + backward + optimize
            # print(f"TEMP INPUTS TYPE: {type(temp_inputs.item())}")
            outputs = net(temp_inputs)
            loss = criterion1(outputs, temp_labels.reshape(net.bz,1).type(torch.float32))
        
            # print("netvhn", net.vhn.weights.shape)
            # print(curly_N(torch.sum(inputs, dim = 0) / dldr.batch_size).shape)
       
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            # print(loss.item())
            # if i % 5 == 4:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
              
            #     running_loss = 0.0
        losses.append(running_loss / len(dldr_trn))
        start = time.time()
        accuracy, aucpr, str_accuracy, str_aucpr = test(net, dldr_tst=dldr_tst)
        end = time.time()
        # print(f"average_inference_time = {end - start}")
        aucpr_tst.append(aucpr)
        inference_times.append((end-start) / len(dldr_tst))

    

    print(f"avg inf time = {sum(inference_times) / num_epochs}")
    return losses, aucpr_tst, arr_epoch
