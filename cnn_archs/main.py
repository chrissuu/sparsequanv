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
from torchprofile import profile_macs

from env_vars import ROOT_LINUX, ROOT_MAC
from BalancingDataset import BalancingDataset
from VHN_conv_net import ATR_A, ATR_B, ATR_C
from preprocess import preprocess_classical, preprocess_resized
from utils import tt_print, tt_print_not_preprocess


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)
LINUX = False
ROOT = ""

NUM_EPOCHS = 1
TEST_SKIPS = 5
BATCH_SIZE = 20
HARDSTOP_TRN = 500
HARDSTOP_TST = 120

if LINUX:

    ROOT = ROOT_LINUX

else:

    ROOT = ROOT_MAC

DATA_TRN_CLASSICAL = f"{ROOT}/sas_nov_data"
DATA_TST_CLASSICAL = f"{ROOT}/sas_june_data"

DATA_TRN_SV_Q = f"{ROOT}/sas_nov_data_qprocessed/8wires"
DATA_TST_SV_Q = f"{ROOT}/sas_june_data_qprocessed/8wires"

DATA_TRN_SV_QS = f"{ROOT}/qnn_data/1000_"
DATA_TST_SV_QS = f"{ROOT}/qnn_data/1000_"

net = None
dldr_trn = None
dldr_tst = None


#############################
#############################
#############################

def CNN_A(bz, data_root_trn, data_root_tst, hardstop_trn, hardstop_tst, num_epochs, test_skips, dldr_trn = None, dldr_tst = None):
    if dldr_trn == None and dldr_tst == None:
        dldr_trn = preprocess_classical(BZ = bz, data_root = data_root_trn, HARDSTOP = hardstop_trn)
        dldr_tst = preprocess_classical(BZ = bz, data_root = data_root_tst, HARDSTOP = hardstop_tst)
        
    net = ATR_A(nc = 1, bz = 20)

    criterion1 = nn.BCELoss()
    criterion2 = None

    optimizer = optim.Adam(net.parameters(), lr=0.0002, betas = (0.5, 0.999))

    configs = (criterion1, criterion2, optimizer, num_epochs, test_skips)
    data = (dldr_trn, dldr_tst)

    _losses, _aucpr_scores, _arr_epoch= tt_print(net, data, configs)

    losses = [_losses[i] for i in range(0, num_epochs, test_skips)]
    arr_epoch = [_arr_epoch[i] for i in range(0, num_epochs, test_skips)]
    vhn_aucpr_tst = [_aucpr_scores[i] for i in range(0, num_epochs, test_skips)]

    total_paramsp = sum(param.numel() for param in net.parameters())
    print(total_paramsp)

    


    return losses, vhn_aucpr_tst, arr_epoch

#############################
#############################
#############################

def CNN_B(bz, data_root_trn, data_root_tst, hardstop_trn, hardstop_tst, num_epochs, test_skips, dldr_trn = None, dldr_tst = None):
    if dldr_trn == None and dldr_tst == None:
        dldr_trn = preprocess_classical(BZ = bz, data_root = data_root_trn, HARDSTOP = hardstop_trn)
        dldr_tst = preprocess_classical(BZ = bz, data_root = data_root_tst, HARDSTOP = hardstop_tst)

    net = ATR_B(nc = 1, bz = 20)

    criterion1 = nn.BCELoss()
    criterion2 = None

    optimizer = optim.Adam(net.parameters(), lr=0.0002, betas = (0.5, 0.999))

    configs = (criterion1, criterion2, optimizer, num_epochs, test_skips)
    data = (dldr_trn, dldr_tst)

    _losses, _aucpr_scores, _arr_epoch= tt_print(net, data, configs)

    losses = [_losses[i] for i in range(0, num_epochs, test_skips)]
    arr_epoch = [_arr_epoch[i] for i in range(0, num_epochs, test_skips)]
    vhn_aucpr_tst = [_aucpr_scores[i] for i in range(0, num_epochs, test_skips)]

    total_paramsp = sum(param.numel() for param in net.parameters())
    print(total_paramsp)

    return losses, vhn_aucpr_tst, arr_epoch


#############################
#############################
#############################

def CNN_C(bz, data_root_trn, data_root_tst, hardstop_trn, hardstop_tst, num_epochs, test_skips):

    dldr_trn = preprocess_resized(BZ = bz, data_root = data_root_trn, HARDSTOP = hardstop_trn)
    dldr_tst = preprocess_resized(BZ = bz, data_root = data_root_tst, HARDSTOP = hardstop_tst)
    net = ATR_C(nc = 1, bz = 20)

    criterion1 = nn.BCELoss()
    criterion2 = None

    optimizer = optim.Adam(net.parameters(), lr=0.0002, betas = (0.5, 0.999))

    configs = (criterion1, criterion2, optimizer, num_epochs, test_skips)
    data = (dldr_trn, dldr_tst)

    _losses, _aucpr_scores, _arr_epoch= tt_print(net, data, configs)

    losses = [_losses[i] for i in range(0, num_epochs, test_skips)]
    arr_epoch = [_arr_epoch[i] for i in range(0, num_epochs, test_skips)]
    vhn_aucpr_tst = [_aucpr_scores[i] for i in range(0, num_epochs, test_skips)]

    total_paramsp = sum(param.numel() for param in net.parameters())
    print(total_paramsp)

    return losses, vhn_aucpr_tst, arr_epoch

    
#############################
#############################
#############################

def CNN_Q(bz, wires, data_trn_sv, data_tst_sv, hardstop_trn, hardstop_tst, num_epochs, test_skips):

    images = np.load(data_trn_sv + "/processed_images.npy")
    labels = np.load(data_trn_sv + "/processed_images_labels.npy")
    dataset_trn = BalancingDataset(image_list = images, label_list= labels, hardstop = hardstop_trn, IR = 1)
    dldr_trn = DataLoader(dataset = dataset_trn, shuffle = True, batch_size = bz)

    images_test = np.load(data_tst_sv + "/processed_images.npy")
    labels_test = np.load(data_tst_sv + "/processed_images_labels.npy")
    dataset_tst = BalancingDataset(image_list = images_test, label_list= labels_test, hardstop = hardstop_tst, IR = 1)
    dldr_tst = DataLoader(dataset = dataset_tst, shuffle = True, batch_size = bz)

    net = ATR_C(nc = wires, bz = bz)

    optimizer = optim.Adam(net.parameters(), lr=0.0002, betas = (0.5, 0.999))
    criterion = nn.BCELoss()
    configs = (criterion, optimizer, num_epochs, test_skips)
    data = (dldr_trn, dldr_tst)
    _losses, _aucpr_scores, _arr_epoch= tt_print_not_preprocess(net, data, configs)

    losses = [_losses[i] for i in range(0, num_epochs, test_skips)]
    arr_epoch = [_arr_epoch[i] for i in range(0, num_epochs, test_skips)]
    aucpr_tst = [_aucpr_scores[i] for i in range(0, num_epochs, test_skips)]

    total_paramsp = sum(param.numel() for param in net.parameters())
    print(total_paramsp)

    return losses, aucpr_tst, arr_epoch

#############################
#############################
#############################

def CNN_QS(bz, wires, data_trn_sv, data_tst_sv, hardstop_trn, hardstop_tst, num_epochs, test_skips):

    images_trn_cmp = np.load(data_trn_sv + "/q_train_dataset.npy")
    labels_trn_cmp = np.load(data_trn_sv + "/q_train_labels.npy")
    dataset_trn_cmp = BalancingDataset(image_list = images_trn_cmp, label_list = labels_trn_cmp, hardstop = hardstop_trn, IR = 1)
    dldr_trn_cmp = DataLoader(dataset = dataset_trn_cmp, shuffle = True, batch_size = bz)

    images_tst_cmp = np.load(data_tst_sv+ "/q_test_dataset.npy")
    labels_tst_cmp = np.load(data_tst_sv + "/q_test_labels.npy")
    dataset_tst_cmp = BalancingDataset(image_list = images_tst_cmp, label_list = labels_tst_cmp, hardstop = hardstop_tst, IR = 1)
    dldr_tst_cmp = DataLoader(dataset = dataset_tst_cmp, shuffle = True, batch_size = bz)

    net_cmp = ATR_C(nc = wires, bz = bz)

    optimizer_cmp = optim.Adam(net_cmp.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    criterion_cmp = nn.BCELoss()
    configs_cmp = (criterion_cmp, optimizer_cmp, num_epochs, test_skips)
    data_cmp = (dldr_trn_cmp, dldr_tst_cmp)
    _losses, _aucpr_scores, _arr_epoch = tt_print_not_preprocess(net_cmp, data_cmp, configs_cmp)

    losses = [_losses[i] for i in range(0, num_epochs, test_skips)]
    arr_epoch = [_arr_epoch[i] for i in range(0, num_epochs, test_skips)]
    aucpr_tst = [_aucpr_scores[i] for i in range(0, num_epochs, test_skips)]

    total_paramsp = sum(param.numel() for param in net_cmp.parameters())
    print(total_paramsp)

    return losses, aucpr_tst, arr_epoch

import matplotlib.pyplot as plt

print("## CNN A ##")
l_a, a_a, e_a = CNN_A(bz = BATCH_SIZE, data_root_trn = DATA_TRN_CLASSICAL, data_root_tst = DATA_TST_CLASSICAL, hardstop_trn = HARDSTOP_TRN, \
                      hardstop_tst = HARDSTOP_TST, num_epochs = NUM_EPOCHS, test_skips = TEST_SKIPS, dldr_trn = dldr_trn, dldr_tst = dldr_tst)

print("## CNN B ##")
l_b, a_b, e_b = CNN_B(bz = BATCH_SIZE, data_root_trn = DATA_TRN_CLASSICAL, data_root_tst = DATA_TST_CLASSICAL, hardstop_trn = HARDSTOP_TRN, \
                      hardstop_tst = HARDSTOP_TST, num_epochs = NUM_EPOCHS, test_skips = TEST_SKIPS, dldr_trn = dldr_trn, dldr_tst = dldr_tst)

print("## CNN C ##")
l_c, a_c, e_c = CNN_C(bz = BATCH_SIZE, data_root_trn = DATA_TRN_CLASSICAL, data_root_tst = DATA_TST_CLASSICAL, hardstop_trn = HARDSTOP_TRN, \
                      hardstop_tst = HARDSTOP_TST, num_epochs = NUM_EPOCHS, test_skips = TEST_SKIPS)

print("## CNN Q ##")
l_q, a_q, e_q = CNN_Q(bz = BATCH_SIZE, wires = 8, data_trn_sv = DATA_TRN_SV_Q, data_tst_sv = DATA_TST_SV_Q, hardstop_trn = HARDSTOP_TRN, \
                      hardstop_tst = HARDSTOP_TST, num_epochs = NUM_EPOCHS, test_skips = TEST_SKIPS)

print("## CNN QS ##")
l_qs, a_qs, e_qs = CNN_QS(bz = BATCH_SIZE, wires = 4, data_trn_sv = DATA_TRN_SV_QS, data_tst_sv = DATA_TST_SV_QS, hardstop_trn = HARDSTOP_TRN, \
                      hardstop_tst = HARDSTOP_TST, num_epochs = NUM_EPOCHS, test_skips = TEST_SKIPS)

plt.plot(e_a, a_a, label='CNN A', linestyle='-', marker=',', color='b')
plt.plot(e_b, a_b, label='CNN B', linestyle='-', marker=',', color='r')
plt.plot(e_c, a_c, label='CNN C', linestyle='-', marker=',', color='g')
plt.plot(e_q, a_q, label='CNN Q', linestyle='-', marker=',', color='y')
plt.plot(e_qs, a_qs, label='CNN QS', linestyle='-', marker=',', color='m')

# Add labels and title
plt.xlabel('Num epochs')
plt.ylabel('AUCPR')
plt.title('Plots of AUCPR with respect to num of epochs')

# print(a_a)
# print(a_b)
# print(a_c)
# print(a_q)
# print(a_qs)

# Add a legend
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()



