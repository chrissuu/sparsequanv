from pennylane import numpy as np
import tensorflow as tf
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pennylane import qml

from Datacompiler import generate_compiler
from utils import *
from train_test import test
from train_test import train
from SparsifiedDataset import QuantumDataset
from SQNN import SQNN

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLOUD = False
LINUX = False
HARDSTOP = 2 # how many imgs to use. 2 * HARDSTOP, balanced
HARDSTOP_TST = 2
BATCH_SIZE = 4 # MAKE SURE BATCH_SIZE ARE FACTORS OF HARDSTOP_TST AND HARDSTO
QUBIT = "lightning.qubit" 
WIRES = 4
# if not LINUX else "lightning.gpu"
path_trn = ""
path_tst = ""
path_hdf = ""

path_vhn_sv = ""
path_vhn_sv_wghts = ""
path_reg_sv = ""
ROOT = ""
SAVE_PATH = "/Users/chrissu/Desktop/research_data/sqnn_data"  # Data saving folder
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

wghts = np.load(path_vhn_sv_wghts)

# dldr_trn_reg, dldr_tst_reg = copy.deepcopy(dldr_trn), copy.deepcopy(dldr_tst)

n_layers = 1   # Number of random layers

np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

#creates iters datasets with skip filters
_rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, WIRES))
np.save(SAVE_PATH + "params", _rand_params)

print("\n\nFINISHED LOADING DATASETS\n\n")
def create_lists(path, path_hdf, BZ, IR, HARDSTOP):
    dldr = generate_compiler(data_root = path, \
                hdf_data_path = path_hdf, \
                BZ = BZ, IR = IR, HARDSTOP = HARDSTOP)
    images = []
    labels = []
    for i, data in enumerate(dldr,0):
        inputs, label = data

        # print(f"type of image: {type(inputs)}\n\n") 
        if i < 2 * HARDSTOP:
            
            if HARDSTOP != float('inf'):
                
                print(f"Loaded {i+1} elt(s)")
            
            else:
                
                if i%50 == 0:
                    print(f"Loaded {i+1} elt(s)")


            inputs = torch.log10(torch.tensor(inputs) + 1)
            inputs = np.array(min_max(inputs))
            # print(inputs.shape)
            images.append(inputs)
            labels.append(label)
        
    return images, labels

train_images, train_labels = create_lists(path = path_trn, path_hdf = path_hdf, BZ = 1, IR = 1, HARDSTOP = HARDSTOP)

test_images, test_labels = create_lists(path = path_tst, path_hdf = path_hdf, BZ = 1, IR = 1, HARDSTOP = HARDSTOP_TST)

dataset_trn_np = np.array(train_images, dtype = 'float')
dataset_tst_np = np.array(test_images, dtype = 'float')

np.save(SAVE_PATH + "q_train_dataset.npy", dataset_trn_np)
np.save(SAVE_PATH + "q_test_dataset.npy", dataset_tst_np)

print(f"Type of dataset: {type(dataset_trn_np)}")
print(f"Shape of TRAIN dataset: {dataset_trn_np.shape}")

dataset_trn = QuantumDataset(dataset_trn_np, np.array(train_labels, dtype = 'float'))
dataset_tst = QuantumDataset(dataset_tst_np, np.array(test_labels, dtype = 'float'))

dldr_trn = DataLoader(dataset_trn, batch_size = BATCH_SIZE, shuffle = True)
dldr_tst = DataLoader(dataset_tst, batch_size = BATCH_SIZE, shuffle  = True)
dev = qml.device(QUBIT, wires=WIRES)


def net():
    return SQNN(bz = BATCH_SIZE, rand_params=_rand_params, q_dev = dev, w=WIRES)

netp = net()
criterion1 = nn.BCELoss()
criterion2 = None
optimizer = optim.Adam(netp.parameters(), lr=0.001)

train(criterion1, criterion2, optimizer, netp, num_epochs=2, dldr_trn = dldr_trn)

print("finished training REG ATR\n")

accuracy, aucpr, str_accuracy, str_aucpr = test(netp, dldr_tst=dldr_tst)

print("finished testing REG ATR\n")

torch.save(netp, './saves/reg.pt')

res_reg_txt = open(path_reg_sv, 'w')
res_reg_txt.write(str_aucpr)
res_reg_txt.write("\n")
res_reg_txt.write(str_accuracy)
res_reg_txt.close()

