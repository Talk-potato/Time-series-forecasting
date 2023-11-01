import pandas as pd
import torch
from torch.utils.data import Dataset

class Link1d_Dataset(Dataset):
    def __init__(self, csv_file, prev_len, next_step=1, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.prev_len = prev_len
        self.next_step = next_step - 1
        self.is_train = is_train

    def __len__(self):
        return len(self.data) - self.prev_len - self.next_step

    def __getitem__(self, idx):
        prev_data = torch.tensor(self.data.iloc[idx:idx + self.prev_len].values).float().unsqueeze(1)
        if self.is_train:
            next_data = torch.tensor(self.data.iloc[idx + self.prev_len + self.next_step].values).float()
            return (prev_data, next_data)
        else:
            return prev_data