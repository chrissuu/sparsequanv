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

from env_vars import ROOT_LINUX, ROOT_MAC
from BalancingDataset import BalancingDataset
from VHN_conv_net import ATR_A, ATR_B
from preprocess import preprocess
from utils import tt_print


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

NUM_EPOCHS = 50
TEST_SKIPS = 1
BATCH_SIZE = 20
HARDSTOP_TRN = 500
HARDSTOP_TST = 120

if LINUX:

    ROOT = ROOT_LINUX

else:

    ROOT = ROOT_MAC

DATA_TRN = f"{ROOT}/sas_nov_data"
DATA_TST = f"{ROOT}/sas_june_data"
DATA_TRN_SV = f"{ROOT}/sas_nov_data_processed"
DATA_TST_SV = f"{ROOT}/sas_june_data_processed"

net = None
dldr_trn = None
dldr_tst = None

#############################
#############################
#############################

def CNN_A():

    dldr_trn = preprocess(BZ = BATCH_SIZE, data_root = DATA_TRN, data_save = DATA_TRN_SV, HARDSTOP = HARDSTOP_TRN)
    dldr_tst = preprocess(BZ = BATCH_SIZE, data_root = DATA_TST, data_save = DATA_TST_SV, HARDSTOP = HARDSTOP_TST)
    net = ATR_A(nc = 1, bz = 20)

    criterion1 = nn.BCELoss()
    criterion2 = None

    optimizer = optim.Adam(net.parameters(), lr=0.002, betas = (0.9, 0.999))

    configs = (criterion1, criterion2, optimizer, NUM_EPOCHS, TEST_SKIPS)
    data = (dldr_trn, dldr_tst)

    _losses, _aucpr_scores, _arr_epoch= tt_print(net, data, configs)

    losses = [_losses[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
    arr_epoch = [_arr_epoch[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
    vhn_aucpr_tst = [_aucpr_scores[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]

    return losses, vhn_aucpr_tst, arr_epoch

#############################
#############################
#############################
def CNN_B():
        
    dldr_trn_cmp = preprocess(BZ = BATCH_SIZE, data_root = DATA_TRN, data_save = DATA_TRN_SV, HARDSTOP = HARDSTOP_TRN)
    dldr_tst_cmp = preprocess(BZ = BATCH_SIZE, data_root = DATA_TST, data_save = DATA_TST_SV, HARDSTOP = HARDSTOP_TST)
    net_cmp = ATR_B(nc = 1, bz = 20)

    criterion1_cmp = nn.BCELoss()
    criterion2_cmp = None

    optimizer_cmp = optim.Adam(net_cmp.parameters(), lr=0.002, betas = (0.9, 0.999))

    configs_cmp = (criterion1_cmp, criterion2_cmp, optimizer_cmp, NUM_EPOCHS, TEST_SKIPS)
    data_cmp = (dldr_trn_cmp, dldr_tst_cmp)

    _losses_cmp, _aucpr_scores_cmp, _arr_epoch_cmp= tt_print(net_cmp, data_cmp, configs_cmp)

    losses_cmp = [_losses_cmp[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
    arr_epoch_cmp = [_arr_epoch_cmp[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
    vhn_aucpr_tst_cmp = [_aucpr_scores_cmp[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]

    return losses_cmp, vhn_aucpr_tst_cmp, arr_epoch_cmp
#############################
#############################
#############################

import matplotlib.pyplot as plt

plt.plot(arr_epoch, vhn_aucpr_tst, label='AUCPR CNN A', linestyle='-', marker='o', color='b')
plt.plot(arr_epoch, losses, label='Loss CNN A', linestyle='-', marker='o', color='r')
# plt.plot(arr_epoch, vhn_aucpr_tst_cmp, label='AUCPR CNN B', linestyle='-', marker='o', color='y')
# plt.plot(arr_epoch, losses_cmp, label='Loss CNN B', linestyle='-', marker='o', color='g')
# plt.plot(arr_epoch, deep_aucpr_tst, label='deep', linestyle='--', marker='s', color='r')

# Add labels and title
plt.xlabel('Num epochs')
plt.ylabel('AUCPR(b), Loss(r)')
plt.title('Plot of AUCPR and Loss with respect to num of epochs')

# Add a legend
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()



