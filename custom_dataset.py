import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Link1d_Dataset(Dataset):
    def __init__(self, traffic_file, prev_len, next_step=1, get_true=True):
        self.data = pd.read_csv(traffic_file)
        self.prev_len = prev_len
        self.next_step = next_step - 1
        self.get_true = get_true

    def __len__(self):
        return len(self.data) - self.prev_len - self.next_step

    def __getitem__(self, idx):
        prev_data = torch.tensor(self.data.iloc[idx:idx + self.prev_len].values).float().unsqueeze(1)
        if self.get_true:
            next_data = torch.tensor(self.data.iloc[idx + self.prev_len : idx + self.prev_len + self.next_step+1].values).float()
            return (prev_data, next_data)
        else:
            return prev_data
        
class NodeLink2d_Dataset(Dataset):
    def __init__(self, traffic_file, node_file, link_file, prev_len, next_step=1, get_true=True):
        self.node = pd.read_csv(node_file)
        self.link = pd.read_csv(link_file)
        self.data = pd.read_csv(traffic_file)
        self.data = self.data.transpose().assign(F_NODE=self.link['F_NODE'].values, T_NODE=self.link['T_NODE'].values).pivot(index=['F_NODE'], columns=['T_NODE'], values=self.data.index.values).stack(0).swaplevel().sort_index().fillna(0).values.reshape((len(self.data), len(self.node), len(self.node)))
        self.prev_len = prev_len
        self.next_step = next_step - 1
        self.get_true = get_true
        
    def __len__(self):
        return len(self.data) - self.prev_len - self.next_step

    def __getitem__(self, idx):
        #data = self.data.iloc[[i+idx for i in range(self.prev_len)]+[idx + self.prev_len + self.next_step]].transpose().assign(F_NODE=self.link['F_NODE'].values, T_NODE=self.link['T_NODE'].values).pivot(index=['F_NODE'], columns=['T_NODE'], values=[i+idx for i in range(self.prev_len+1)]).stack(0).swaplevel().sort_index().fillna(0).values.reshape((self.prev_len+1, len(self.node), len(self.node)))
        prev_data = self.data[idx:idx+self.prev_len]
        prev_data = torch.tensor(prev_data).float()
        if self.get_true:
            next_data = torch.tensor(self.data[idx+self.prev_len+self.next_step]).float()
            return (prev_data, next_data)
        else:
            return prev_data
        
class Link_Time_Dataset(Dataset):
    def __init__(self, traffic_file, time_file, prev_len, next_step=1, get_true=True):
        self.data = pd.read_csv(traffic_file)
        self.time = pd.read_csv(time_file)
        self.prev_len = prev_len
        self.next_step = next_step - 1
        self.get_true = get_true

    def __len__(self):
        return len(self.data) - self.prev_len - self.next_step

    def __getitem__(self, idx):
        prev_data = torch.tensor(self.data.iloc[idx:idx + self.prev_len].values).float().unsqueeze(1)
        prev_time = torch.tensor(self.time.iloc[idx:idx + self.prev_len].values)
        if self.get_true:
            next_data = torch.tensor(self.data.iloc[idx + self.prev_len + self.next_step].values).float()
            next_time = torch.tensor(self.time.iloc[idx + self.prev_len + self.next_step].values)
            return (prev_data, prev_time, next_data, next_data)
        else:
            return prev_data, prev_time
        
