from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from Dataloader import generate_generators

dldr = generate_generators(data_root='../../../../research_data/sas_nov_data/', hdf_data_path='DL_info/chip_info/cube_raw', BZ = 20, IR = 1)

vhn_wght = np.load('./saves/vhn_wghts.npy')

save_data = []

for i, data in enumerate(dldr, 0):
    
    inputs, labels = data

    inputs = torch.log10(torch.tensor(inputs).transpose(1,3)+1)

    save_data.append(inputs)



batch_pull_number = 10 # which part from the batch to pull from, 0 <= x < 20



temp = save_data[batch_pull_number].numpy()
f = f_VHN(torch.tensor(temp), torch.tensor(vhn_wght))
c = plt.imshow(np.array(f[0][50]))
    
plt.colorbar(c)
plt.show()
