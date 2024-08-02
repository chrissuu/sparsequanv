import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import binary_auprc
import matplotlib.pyplot as plt
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
from env_vars import ROOT_LINUX, ROOT_MAC
from QKATR import ATR
from preprocess import preprocess, preprocess_resized
from utils import tt_print, tt_print_not_preprocess


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


NUM_EPOCHS = 200
TEST_SKIPS = 5
BATCH_SIZE = 20
WIRES = 8
HARDSTOP_TRN = 500
HARDSTOP_TST = 120
NLAYERS = 1
QUBIT = "lightning.qubit" 
RPARAMS = rand_params = np.random.uniform(high=2 * np.pi, size=(NLAYERS, WIRES))

if LINUX:

    ROOT = ROOT_LINUX

else:

    ROOT = ROOT_MAC

DATA_TRN = f"{ROOT}/sas_nov_data"
DATA_TST = f"{ROOT}/sas_june_data"
DATA_TRN_SV = f"{ROOT}/sas_nov_data_qprocessed/8wires"
DATA_TST_SV = f"{ROOT}/sas_june_data_qprocessed/8wires"

DATA_TRN_SV_CMP = f"{ROOT}/qnn_data/1000_"
DATA_TST_SV_CMP = f"{ROOT}/qnn_data/1000_"

net = None
dldr_trn = None
dldr_tst = None

net_cmp = None
dldr_trn_cmp = None
dldr_tst_cmp = None

dev = qml.device(QUBIT, wires=WIRES)

images = np.load(DATA_TRN_SV + "/processed_images.npy")
labels = np.load(DATA_TRN_SV + "/processed_images_labels.npy")
dataset_trn = BalancingDataset(image_list = images, label_list= labels, hardstop = HARDSTOP_TRN, IR = 1)
dldr_trn = DataLoader(dataset = dataset_trn, shuffle = True, batch_size = BATCH_SIZE)

images_test = np.load(DATA_TST_SV + "/processed_images.npy")
labels_test = np.load(DATA_TST_SV + "/processed_images_labels.npy")
dataset_tst = BalancingDataset(image_list = images_test, label_list= labels_test, hardstop = HARDSTOP_TST, IR = 1)
dldr_tst = DataLoader(dataset = dataset_tst, shuffle = True, batch_size = BATCH_SIZE)

net = ATR(nc = WIRES, bz = BATCH_SIZE)

optimizer = optim.Adam(net.parameters(), lr=0.0002, betas = (0.5, 0.999))
criterion = nn.BCELoss()
configs = (criterion, optimizer, NUM_EPOCHS, TEST_SKIPS)
data = (dldr_trn, dldr_tst)
_losses, _aucpr_scores, _arr_epoch= tt_print_not_preprocess(net, data, configs)

####################


images_trn_cmp = np.load(DATA_TRN_SV_CMP + "/q_train_dataset.npy")
labels_trn_cmp = np.load(DATA_TRN_SV_CMP + "/q_train_labels.npy")
dataset_trn_cmp = BalancingDataset(image_list = images_trn_cmp, label_list = labels_trn_cmp, hardstop = HARDSTOP_TRN, IR = 1)
dldr_trn_cmp = DataLoader(dataset = dataset_trn_cmp, shuffle = True, batch_size = BATCH_SIZE)

images_tst_cmp = np.load(DATA_TST_SV_CMP + "/q_test_dataset.npy")
labels_tst_cmp = np.load(DATA_TST_SV_CMP + "/q_test_labels.npy")
dataset_tst_cmp = BalancingDataset(image_list = images_tst_cmp, label_list = labels_tst_cmp, hardstop = HARDSTOP_TST, IR = 1)
dldr_tst_cmp = DataLoader(dataset = dataset_tst_cmp, shuffle = True, batch_size = BATCH_SIZE)

net_cmp = ATR(nc = int(WIRES/2), bz = BATCH_SIZE)

optimizer_cmp = optim.Adam(net_cmp.parameters(), lr = 0.0002, betas = (0.5, 0.999))
criterion_cmp = nn.BCELoss()
configs_cmp = (criterion_cmp, optimizer_cmp, NUM_EPOCHS, TEST_SKIPS)
data_cmp = (dldr_trn_cmp, dldr_tst_cmp)
_losses_cmp, _aucpr_scores_cmp, _arr_epoch_cmp = tt_print_not_preprocess(net_cmp, data_cmp, configs_cmp)


losses = [_losses[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
arr_epoch = [_arr_epoch[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
aucpr_tst = [_aucpr_scores[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]

losses_cmp = [_losses_cmp[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
arr_epoch_cmp = [_arr_epoch_cmp[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]
aucpr_tst_cmp = [_aucpr_scores_cmp[i] for i in range(0, NUM_EPOCHS, TEST_SKIPS)]

# F = open("../saves/8wires_q.txt", "w")
# F.write(f"{arr_epoch}${losses}${aucpr_tst}")

import matplotlib.pyplot as plt

plt.plot(arr_epoch, aucpr_tst, label='AUCPR CNN Q', linestyle='-', marker='o', color='b')
plt.plot(arr_epoch, losses, label='Loss CNN Q', linestyle='-', marker='o', color='r')
plt.plot(arr_epoch_cmp, aucpr_tst_cmp, label = 'AUCPR CNN QS', linestyle='-', marker = 'o', color= 'y')
plt.plot(arr_epoch_cmp, losses_cmp, label = 'Loss CNN QS', linestyle = '-', marker = 'o', color = 'g')

# Add labels and title
plt.xlabel('Num Epochs')
plt.ylabel('AUCPR, Loss')
plt.title('AUCPR and Loss of CNN Q and CNN QS w.r.t Number of Training Epochs')

# Add a legend
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()

#inftime = 0.17203134546677262
#inftime = 0.05

