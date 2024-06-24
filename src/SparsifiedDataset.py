from torch.utils.data import Dataset

class QuantumDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.dataset_len = len(image_list)
        self.image_list = image_list
        self.label_list = label_list
        pass

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        return (self.image_list[idx], self.label_list[idx])
