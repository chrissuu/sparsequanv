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
from pennylane.templates import RandomLayers


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLOUD = False
LINUX = False
HARDSTOP = 128 # how many imgs to use. 2 * HARDSTOP, balanced
HARDSTOP_TST = float('inf')
IMB_RAT = 2
BATCH_SIZE = 4 # MAKE SURE BATCH_SIZE ARE FACTORS OF HARDSTOP_TST AND HARDSTO
QUBIT = "lightning.qubit" 
WIRES = 8
# if not LINUX else "lightning.gpu"
path_trn = ""
path_tst = ""
path_hdf = ""

path_vhn_sv = ""
path_vhn_sv_wghts = ""
path_reg_sv = ""
ROOT = ""
SAVE_PATH = "/Users/chrissu/Desktop/research_data/sqnn_data/"  # Data saving folder
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


#creates iters datasets with skip filters
def generate_datum(train_images, n_layers, iters, skip):
    
    train_datasets = [[[] for img in train_images] for i in range(skip)]
    stores = [[] for i in range(len(train_images))]
    
        
    dev = qml.device(QUBIT, wires=WIRES)
    # Random circuit parameters
    

    @qml.qnode(dev)
    def circuit(phi, rand_params):
        # Encoding of 8 classical input values
        for j in range(WIRES):
            qml.RY(np.pi * phi[j], wires=j)
        # print("qml.RY(np.pi * phi[j], wires = j") 
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(WIRES)))
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(WIRES)]

    #produces 4 kerneled images of size 14,14 per image
    def quanv(image, rand_params):
        """Convolves the input image with many applications of the same quantum circuit."""
        tot = np.zeros((32, 32, 50, WIRES))
        
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, 64, 2):
            for k in range(0, 64, 2):
                for m in range(0, 100, 2):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    
                    q_results = circuit(
                        [   
                            image[j, k, m],
                            image[j+1, k, m],
                            image[j, k+1, m],
                            image[j+1, k+1, m],
                            image[j, k, m+1],
                            image[j+1, k, m+1],
                            image[j, k+1, m+1],
                            image[j+1, k+1, m+1],                                        
                        ],
                        rand_params
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(WIRES):
                        tot[j // 2, k // 2, m //2, c] = q_results[c]
        return tot
    
    start = time.time()
    for i in range(skip * iters): 
        # for skip*iters iterations, create a random circuit 
        # and use that circuit to generate 4 kerneled images
        # when this loop exits, there would be skip * iters * WIRES kerneled images,
        # imgs, in stores[imgs]
        print(f"Generating kerneled dataset {i} of filtered images\n")
        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, WIRES))
        for idx, img in enumerate(train_images):
            start_temp = time.time()
            stores[idx].append(quanv(img, rand_params))
            end_temp = time.time()
            print(f"img {idx+1} took {end_temp - start_temp} seconds to process")
    end = time.time()
    print(f"Finished generating filtered images. this task took {end - start} seconds. \n\nEntering dataset generation\n\n")
   
    # from the previous loop, we now want to put these imgs in datasets
    # i loops over the number of datasets (iters)

    
    for i in range(0, skip):
        
        # for each img array in stores, 
        # enumerating by idx, we take a subarray of size i * skip, copy, then reshape to desired size
        # appropriate data with appropriate idx is added
        for idx, img_array in enumerate(stores):
            temp = np.array(img_array[i: i+1], dtype = 'float').copy().reshape((32, 32, 50, WIRES))
            train_datasets[i][idx].append(temp)
    
    print("Finished dataset generation\n\n")
    return train_datasets

def generate_data(images, num_filters, n_layers):
    return generate_datum(train_images=images, iters = 1, skip = num_filters, n_layers = n_layers)


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
                
                print(f"Loaded {i+1} elt(s)")
            
            else:
                
                if i%50 == 0:
                    print(f"Loaded {i+50} elt(s)")


            inputs = torch.log10(torch.tensor(inputs) + 1)
            inputs = np.array(min_max(inputs))
            # print(inputs.shape)
            images.append(inputs)
            labels.append(label)
            
    print(f"There were ({target_cnt} targets), ({len(images) - target_cnt} clutters) in this dataset of size {len(images)}, of IR = {IMB_RAT}")
    return images, labels

def generate_trn_tst(SAVE_PATH, path_trn, path_tst, path_hdf, BZ, IR, HARDSTOP, HARDSTOP_TST):
    train_images, train_labels = create_lists(path = path_trn, path_hdf = path_hdf, BZ = BZ, IR = IR, HARDSTOP = HARDSTOP)

    test_images, test_labels = create_lists(path = path_tst, path_hdf = path_hdf, BZ = BZ, IR = IR, HARDSTOP = HARDSTOP_TST)

    np.save(SAVE_PATH + "q_train_images.npy", train_images)
    np.save(SAVE_PATH + "q_test_images.npy", test_images)
    np.save(SAVE_PATH + "q_train_labels.npy", train_labels)
    np.save(SAVE_PATH + "q_test_labels.npy", test_labels)

    print("\n\nFINISHED LOADING DATASETS\n\n")

    print(f"Len of TRAIN dataset: {len(train_images)}")
    print(f"Len of TEST dataset: {len(test_images)}")
    print(f"Shape of data: {test_images[0].shape}")
    # shape shold be 20, 101, 64, 64 before CNN input.#

    skip = 1 #skips ~ num filters. #kerneled imgs = skip * 8
    iters = 1 #only change iters if you want to test scaling laws
    n_layers = 1
    dataset_trn_np = np.array(generate_data(images=train_images, n_layers = n_layers, num_filters = skip)[0], dtype = 'float').squeeze(1)
    dataset_tst_np = np.array(generate_data(images=test_images, n_layers = n_layers, num_filters = skip)[0], dtype = 'float').squeeze(1)

    np.save(SAVE_PATH + "q_train_dataset.npy", dataset_trn_np)
    np.save(SAVE_PATH + "q_test_dataset.npy", dataset_tst_np)

    return dataset_trn_np, dataset_tst_np
