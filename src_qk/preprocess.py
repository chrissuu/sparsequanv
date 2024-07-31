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

from env_vars import ROOT_LINUX, ROOT_MAC
from BalancingDataset import BalancingDataset
from utils import min_max

LINUX = False
ROOT = ""

if LINUX:

    ROOT = ROOT_LINUX

else:

    ROOT + ROOT_MAC

DATA_TRN = f"{ROOT}/sas_nov_data_qprocessed/"
DATA_TST = f"{ROOT}/sas_june_data_qprocessed/"

def circuit(WIRES, rand_params, phi):
        # Encoding of KERNEL classical input values
        for j in range(WIRES):
            qml.RY(np.pi * phi[j], wires=j)
        # print("qml.RY(np.pi * phi[j], wires = j") 
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(WIRES)))
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(WIRES)]

# produces WIRES kerneled images of size 14,14 per image
def quanv_sparse(w, l, h, kw, kl, kh, WIRES, image, dev, rand_params):
    if w % kw == 1:
        w -= 1

    if l % kl == 1:
        l -= 1

    if h % kh == 1:
        # print("HERE")
        h -= 1
    tot = np.zeros((w // kw, l // kl, h // kh, WIRES))
    print(image.shape)
    image = np.squeeze(image, axis = 0)
    # Loop over the coordinates of the top-left pixel of dim kernel_dim^3 squares
    start = time.time()
    
    for j in range(0, l, kl):
        for k in range(0, w, kw):
            for m in range(0, h, kh):
                # Process a squared 2x2 region of the image with a quantum circuit
                # q_results = circuit(
                #     [
                #         image[j, k, m],
                #         image[j + 1, k + 1, m],
                #         image[j, k + 1, m+1],
                #         image[j + 1, k, m+1],
                #     ]
                # )
                qnode = qml.QNode(circuit,dev)
                # (im_l, im_w, im_h) = image.shape
                
                # assert(j < im_l)
                # assert(k < im_w)
                # assert(m < im_h)
                # print(image.shape)
                q_results = None
                if WIRES == 4:
                    q_results = qnode(
                        WIRES,
                        rand_params,
                        [
                            image[j, k, m],
                            image[j + 1, k + 1, m],
                            image[j, k + 1, m+1],
                            image[j + 1, k, m+1],
                        ]
                    )
                else:
                    q_results = qnode(
                        WIRES,
                        rand_params,
                        [   
                            image[j, k, m],
                            image[j+1, k, m],
                            image[j, k+1, m],
                            image[j+1, k+1, m],
                            image[j, k, m+1],
                            image[j+1, k, m+1],
                            image[j, k+1, m+1],
                            image[j+1, k+1, m+1],                                        
                        ]
                    )
                
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(WIRES):
                    # print(q_results[0])
                    tot[j // kl, k // kw, m // kh, c] = q_results[c]
            # print(WIRES)
    end = time.time()

    if print:

        print(f"Image processing for sparse quanv took {end-start} seconds")
    return tot

def readHDF(file_name, hdf_path, data_root):
    """ Reads data from the HDF"""
    with h5py.File(os.path.join(data_root, file_name), mode='r') as f:
        data = f[hdf_path][:]
        data = np.transpose(data)  # because python reverses the order of the 3d volume, this will correct back to the original matlab order


    return data

def chip_center(data, batch_size):
    HDF_n_rows = 71
    HDF_n_cols = 71
    HDF_n_dpth = 101

    input_shape = (batch_size, HDF_n_rows, HDF_n_cols, HDF_n_dpth)

    chip_n_rows = 64
    chip_n_cols = 64
    chip_n_dpth = 101

    "chip shape is one cube from batch"
    chip_shape = (chip_n_rows, chip_n_cols, chip_n_dpth)

    # Make a boolean indexing mask for efficient extraction of the chip center:
    "input shape minus chip shape row"
    row_diff = input_shape[1] - chip_shape[0]
    "this is chopping off one half the difference on each side"
    first_row = row_diff // 2
    last_row = first_row + chip_shape[0]

    "this is chopping off one half the difference on each side"
    col_diff = input_shape[2] - chip_shape[1]
    first_col = col_diff // 2
    last_col = first_col + chip_shape[1]

    "this is chopping off one half the difference on each side"
    slice_diff = input_shape[3] - chip_shape[2]
    first_slice = slice_diff // 2
    last_slice = first_slice + chip_shape[2]

    "slicing out the chips from the input data"
    "list with shape of input all False values"
    center_select = np.full((HDF_n_rows, HDF_n_cols, HDF_n_dpth), False)
    "except the chips size all true"
    center_select[first_row:last_row, first_col:last_col, first_slice:last_slice] = True

    # first_slice = 0
    # last_slice = bsz_by_class

        
    """ extracts the center of the chip via boolean indexing. """
    return np.reshape(data[center_select], chip_shape)

def preprocess(dev, data_root, data_save, HARDSTOP, WIRES, BZ, rand_params, hdf_data_path = 'DL_info/chip_info/cube_raw', IR = 1):
    n_classes = 2
    file_list = [[] for clas in range(n_classes)]
    
    #print('Allocating HDFs to train/valid/test...')
    
    for filename in os.listdir(data_root):
        if not filename.endswith('.hdf'):
            continue

        # Extract the label using 'label_scheme' identifier
        label = int(int(filename.split('_')[3]) > 0)  # 0 = clutter (not manmade); 1 = target (manmade)   
        file_list[label].append(filename)

    print(f"clutter len: {len(file_list[0])}")
    print(f"target len: {len(file_list[1])}")     

    data_root = data_root
    n_classes = n_classes

    "balance by class"
    hdf_path = hdf_data_path
    batch_size = BZ
    bsz_by_class = int(batch_size // 2)
    chip_n_rows = 64
    chip_n_cols = 64
    chip_n_dpth = 101
    
    chip_shape = (chip_n_rows, chip_n_cols, chip_n_dpth)
    
    images_list = []
    labels_list = []
    

    for file in file_list[0]:
        data = readHDF(file_name = file, hdf_path=hdf_path, data_root = data_root)
        data = chip_center(data, batch_size = BZ)
        images_list.append(data)
        labels_list.append(0)

    for file in file_list[1]:
        data = readHDF(file_name = file, hdf_path=hdf_path, data_root = data_root)
        data = chip_center(data, batch_size = BZ)
        images_list.append(data)
        labels_list.append(1)
    print(images_list[0].shape)
    # for label in labels_list:
    #     print (label)
    # # TESTED UP TO HERE
    # print(images_list)

    dataset = BalancingDataset(image_list = images_list, label_list = labels_list, hardstop= HARDSTOP, IR=IR)
    dldr = DataLoader(dataset = dataset, batch_size = 1, shuffle = True)
    images = []
    labels = []
    
    # print(labels)

    np.save(f"{data_save}/processed_images_raw",np.array(images))
    
    for i, data in enumerate(dldr):
        # print("HERE")
        inputs, label = data

        # print(f"type of image: {type(inputs)}\n\n") 
        if i < 2 * HARDSTOP:
            
            if HARDSTOP != float('inf'):
                
                print(f"Loaded {i+1} elt(s)")
            
            else:
                
                if i%50 == 0:
                    print(f"Loaded {i+1} elt(s)")

            inputs = torch.log10(inputs + 1)
            inputs = np.array(min_max(inputs))
            inputs = quanv_sparse(dev = dev,w = chip_n_cols, l = chip_n_rows, h = chip_n_dpth, \
                                  kw = 2, kl = 2, kh = 2, WIRES = WIRES, rand_params = rand_params, image = inputs)
            # print(inputs.shape)
            images.append(inputs)
            labels.append(label)

    # print(len(images))
    # print(len(labels))
    np.save(f"{data_save}/processed_images",np.array(images))
    np.save(f"{data_save}/processed_images_labels",np.array(labels))
    print(len(images))
    print(len(labels))
    dataset_res = BalancingDataset(image_list = images, label_list = labels, hardstop = HARDSTOP, IR = IR)
    dldr_ret = DataLoader(dataset = dataset_res, shuffle = True, batch_size = BZ)
    return dldr_ret



def preprocess_resized(data_root, data_save, HARDSTOP, BZ, hdf_data_path = 'DL_info/chip_info/cube_raw', IR = 1):
    n_classes = 2
    file_list = [[] for clas in range(n_classes)]
    
    #print('Allocating HDFs to train/valid/test...')
    
    for filename in os.listdir(data_root):
        if not filename.endswith('.hdf'):
            continue

        # Extract the label using 'label_scheme' identifier
        label = int(int(filename.split('_')[3]) > 0)  # 0 = clutter (not manmade); 1 = target (manmade)   
        file_list[label].append(filename)

    print(f"clutter len: {len(file_list[0])}")
    print(f"target len: {len(file_list[1])}")     

    data_root = data_root
    n_classes = n_classes

    "balance by class"
    hdf_path = hdf_data_path
    batch_size = BZ
    bsz_by_class = int(batch_size // 2)
    chip_n_rows = 64
    chip_n_cols = 64
    chip_n_dpth = 101
    
    chip_shape = (chip_n_rows, chip_n_cols, chip_n_dpth)
    
    images_list = []
    labels_list = []
    

    for file in file_list[0]:
        data = readHDF(file_name = file, hdf_path=hdf_path, data_root = data_root)
        data = chip_center(data, batch_size = BZ)
        images_list.append(data)
        labels_list.append(0)

    for file in file_list[1]:
        data = readHDF(file_name = file, hdf_path=hdf_path, data_root = data_root)
        data = chip_center(data, batch_size = BZ)
        images_list.append(data)
        labels_list.append(1)
    # print(images_list[0].shape)
    # for label in labels_list:
    #     print (label)
    # # TESTED UP TO HERE
    # print(images_list)

    dataset = BalancingDataset(image_list = images_list, label_list = labels_list, hardstop= HARDSTOP, IR=IR)
    dldr = DataLoader(dataset = dataset, batch_size = 1, shuffle = True)
    images = []
    labels = []
    
    # print(labels)

    
    for i, data in enumerate(dldr):
        # print("HERE")
        inputs, label = data

        # print(f"type of image: {type(inputs)}\n\n") 
        if i < 2 * HARDSTOP:
            
            if HARDSTOP != float('inf'):
                
                print(f"Loaded {i+1} elt(s)")
            
            else:
                
                if i%50 == 0:
                    print(f"Loaded {i+1} elt(s)")

            inputs = torch.log10(inputs + 1)
            inputs = np.array(min_max(inputs))
            # print(inputs.shape)
            inputs = np.resize(inputs, (1, 32, 32, 50))
            # print(inputs.shape)
            images.append(inputs)
            labels.append(label)

    # print(len(images))
    # print(len(labels))
    # print(len(images))
    # print(len(labels))
    # print(images[0].shape)
    dataset_res = BalancingDataset(image_list = images, label_list = labels, hardstop = HARDSTOP, IR = IR)
    dldr_ret = DataLoader(dataset = dataset_res, shuffle = True, batch_size = BZ)
    return dldr_ret
