from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

def cnt(label_list):
    target_cnt = 0
    clutter_cnt = 0

    for i in range(len(label_list)):
        if label_list[i] == 0:
            clutter_cnt += 1
        else:
            target_cnt += 1

    return target_cnt, clutter_cnt
def balance(image_list, label_list, hardstop, IR):
    target_cnt, clutter_cnt = cnt(label_list)

    assert(hardstop == float('inf') or (hardstop <= target_cnt and hardstop * IR <= clutter_cnt))

    image_list_temp = [None for i in range(hardstop*(1+IR))]
    label_list_temp = [None for i in range(hardstop*(1+IR))]
    
    _info = {0: 0,
             1: 0}
    
    for i in range(len(image_list)):
        label = label_list[i]
        # print(label)
        label = int(label)
        if label == 0 and _info[label] < hardstop * IR:
            _info[label] += 1
            image_list_temp[i] = image_list[i]
            label_list_temp[i] = np.array(0, dtype='float')
        elif label == 1 and _info[label] < hardstop:
            _info[label] += 1
            image_list_temp[i] = image_list[i]
            label_list_temp[i] = np.array(1, dtype = 'float')
        else:
            continue
    # print(f"LENNNNN {len(image_list_temp)}")
    return image_list_temp, label_list_temp

        

class QuantumDataset(Dataset):
    def __init__(self, image_list, label_list, hardstop, IR = 1):
        self.dataset_len = (1+IR) * hardstop
        # print(hardstop)
        self.image_list, self.label_list = balance(image_list, label_list, hardstop, IR)
        pass

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        return (self.image_list[idx], self.label_list[idx])