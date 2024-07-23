from pennylane import numpy as np
import tensorflow as tf
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Datacompiler import generate_compiler
from utils import *
from train_test import train_print
from SparsifiedDataset import QuantumDataset, BalancedBatchSampler
from VHN_conv_net import ATR

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLOUD = False
LINUX = False
HARDSTOP = 500 # how many imgs to use. 2 * HARDSTOP, balanced
HARDSTOP_TST = 120
IMB_RAT = 1
BATCH_SIZE = 20 # MAKE SURE BATCH_SIZE ARE FACTORS OF HARDSTOP_TST AND HARDSTO
# if not LINUX else "lightning.gpu"
path_trn = ""
path_tst = ""
path_hdf = ""

path_vhn_sv = ""
path_vhn_sv_wghts = ""
path_reg_sv = ""
ROOT = ""
SAVE_PATH = "/Users/chrissu/Desktop/research_data/classical_data/"  # Data saving folder
if CLOUD:
    path_trn = '/data/sjayasur/greg_data/train/'
    path_tst = '/data/sjayasur/greg_data/test/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts = '/home/chrissu/saves/vhn_weights.txt'
    path_vhn_sv = '/home/chrissu/saves/res_vhn.txt'
    path_reg_sv = '/home/chrissu/saves/res_reg.txt'
elif LINUX:
    path_trn = '/home/imaginglyceum/Desktop/research_data/sas_nov_data/'
    path_tst = '/home/imaginglyceum/Desktop/research_data/sas_june_data/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts = '/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline_test/saves/vhn_wghts.npy'
    path_vhn_sv = '/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline_test/saves/res_vhn.txt'
    path_reg_sv = '/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline_test/saves/res_reg.txt'
else:
    ROOT = "/Users/chrissu/Desktop"
    path_trn = f'{ROOT}/research_data/sas_nov_data/'
    path_tst = f'{ROOT}/research_data/sas_june_data/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts =f'{ROOT}/reu_suren_lab2024/research/scripts/pipeline/saves/vhn_wghts.npy'
    path_vhn_sv = './saves/res_vhn.txt'
    path_reg_sv = './saves/res_reg.txt'

def create_lists(path, path_hdf, BZ, IR, HARDSTOP):
    dldr = generate_compiler(data_root = path, \
                hdf_data_path = path_hdf, \
                BZ = BZ, IR = IR, HARDSTOP = HARDSTOP)
    images = []
    labels = []

    target_cnt = 0
    for i, data in enumerate(dldr,0):
        inputs, label = data
        # print(f"type of image: {type(inputs)}\n\n") 
        if i < (IR + 1) * HARDSTOP:
            if int(label) == 1:
                target_cnt += 1

            if HARDSTOP != float('inf'):
                
                if i%50 == 0:
                    print(f"Loaded {i+50} elt(s)")
            
            else:
                
                if i%50 == 0:
                    print(f"Loaded {i+50} elt(s)")


            

            # print(inputs.shape)
            images.append(inputs)
            labels.append(label)
            
    print(f"There were ({target_cnt} targets), ({len(images) - target_cnt} clutters) in this dataset of size {len(images)}, of IR = {IMB_RAT}")
    return images, labels

train_images, train_labels = create_lists(path = path_trn, path_hdf = path_hdf, BZ = 1, IR = IMB_RAT, HARDSTOP = HARDSTOP)

test_images, test_labels = create_lists(path = path_tst, path_hdf = path_hdf, BZ = 1, IR = IMB_RAT, HARDSTOP = HARDSTOP_TST)

dataset_trn_np = np.array(train_images, dtype = 'float')
dataset_tst_np = np.array(test_images, dtype = 'float')

np.save(SAVE_PATH + "q_train_dataset.npy", dataset_trn_np)
np.save(SAVE_PATH + "q_test_dataset.npy", dataset_tst_np)

print(f"Type of dataset: {type(dataset_trn_np)}")
print(f"Shape of TRAIN dataset: {dataset_trn_np.shape}")

dataset_trn = QuantumDataset(dataset_trn_np, np.array(train_labels, dtype = 'float'), hardstop = HARDSTOP, IR = IMB_RAT)
dataset_tst = QuantumDataset(dataset_tst_np, np.array(test_labels, dtype = 'float'), hardstop = HARDSTOP_TST, IR = IMB_RAT)


# balanced_sampler_trn = BalancedBatchSampler(dataset_trn, BATCH_SIZE)
# balanced_sampler_tst = BalancedBatchSampler(dataset_tst, BATCH_SIZE)

# dldr_trn = DataLoader(dataset_trn, batch_sampler=balanced_sampler_trn)
# dldr_tst = DataLoader(dataset_tst, batch_sampler=balanced_sampler_tst)

dldr_trn = DataLoader(dataset_trn, batch_size = BATCH_SIZE, shuffle = True)
dldr_tst = DataLoader(dataset_tst, batch_size = BATCH_SIZE, shuffle = True )

NUM_EPOCHS = 100

net_vhn = ATR(nc = 1, bz = BATCH_SIZE) # initializes VHN convnet; nc = input should have 1 channel
criterion1 = nn.BCELoss()
criterion2 = None
optimizer = optim.Adam(net_vhn.parameters(), lr=0.0004)

losses, arr_epoch, vhn_aucpr_tst = train_print(criterion1, criterion2, optimizer, net_vhn, num_epochs = NUM_EPOCHS, dldr_trn = dldr_trn, dldr_tst = dldr_tst)

plt.plot(arr_epoch, vhn_aucpr_tst, label='aucpr', linestyle='--', marker='s', color='y')
plt.plot(arr_epoch, losses, label='losses', linestyle='--', marker='s', color='b')

plt.xlabel('Num epochs')
plt.ylabel('ATRP (Blue), SQNN (Red)')
plt.title('Plot of AUCPR of atrp and sqnn with respect to num of epochs')

plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()

