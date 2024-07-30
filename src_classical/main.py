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
from VHN_conv_net import ATR
from preprocess import preprocess
from utils import tt_print


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

LINUX = False
ROOT = ""
AS_PREPROCESSING = True


NUM_EPOCHS = 100
TEST_SKIPS = 5
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

if AS_PREPROCESSING:
    
    dldr_trn = preprocess(BZ = BATCH_SIZE, data_root = DATA_TRN, data_save = DATA_TRN_SV, HARDSTOP = HARDSTOP_TRN)
    dldr_tst = preprocess(BZ = BATCH_SIZE, data_root = DATA_TST, data_save = DATA_TST_SV, HARDSTOP = HARDSTOP_TST)
    net = ATR(nc = 1, bz = 20)

criterion1 = nn.BCELoss()
criterion2 = None

optimizer = optim.Adam(net.parameters(), lr=0.0002, betas = (0.9, 0.999))

configs = (criterion1, criterion2, optimizer, NUM_EPOCHS, TEST_SKIPS)
data = (dldr_trn, dldr_tst)

_losses, _aucpr_scores, _arr_epoch= tt_print(net, data, configs)

losses = [_losses[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
arr_epoch = [_arr_epoch[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
vhn_aucpr_tst = [_aucpr_scores[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]

import matplotlib.pyplot as plt

plt.plot(arr_epoch, vhn_aucpr_tst, label='aucpr', linestyle='-', marker='o', color='b')
plt.plot(arr_epoch, losses, label='loss', linestyle='-', marker='o', color='r')
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



