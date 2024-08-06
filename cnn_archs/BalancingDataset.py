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
    
    hardstop = target_cnt if hardstop == float('inf') else hardstop
    assert(hardstop <= target_cnt and hardstop * IR <= clutter_cnt)

    _info = {0: 0,
             1: 0}
    # print(f"IR {IR}")
    # print(f"TARGET CNT: {target_cnt}, CLUTTER CNT: {clutter_cnt}")
    target_list_temp = []
    clutter_list_temp = []
    t_label_list_temp = []
    c_label_list_temp = []

    for i in range(len(image_list)):
        if label_list[i] == 1:
            target_list_temp.append(image_list[i])
            t_label_list_temp.append(1)

    for i in range(len(image_list)):
        if label_list[i] == 0:
            clutter_list_temp.append(image_list[i])
            c_label_list_temp.append(0)

    ret_img_list = []
    ret_label_list = []

    for i in range(hardstop):
        ret_img_list.append(target_list_temp[i])
        ret_label_list.append(1)
    for i in range(hardstop * IR):
        ret_img_list.append(clutter_list_temp[i])
        ret_label_list.append(0)
    return ret_img_list, ret_label_list

class BalancingDataset(Dataset):
    def __init__(self, image_list, label_list, hardstop, IR = 1):
        # print(f"HARDSTOP: {hardstop}")
        self.dataset_len = len(image_list) if hardstop == float('inf') else (1+IR) * hardstop 
        # print(hardstop)
        self.image_list, self.label_list = balance(image_list, label_list, hardstop, IR)
        # print(f"POST IMG LEN: {len(self.image_list)}, POST LABEL LEN: {len(self.label_list)}")
        pass

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return (self.image_list[idx], self.label_list[idx])
    
