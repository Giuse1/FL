from torch.utils.data import Dataset
import torch
import numpy as np
import ast
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

import os


class ClientDataset(Dataset):

    def __init__(self, path, transform=None):

        self.path = path
        self.transform = transform
        self.df = pd.read_csv(path)


    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.df.loc[idx, 'labels'])
        input = np.array(ast.literal_eval(self.df.loc[idx,'pixels'])).reshape(28, 28)
        if self.transform:
            input = self.transform(input)
        sample = {'input': input, 'label': label}

        return sample


class ValidationDataset(Dataset):

    def __init__(self, path, transform=None):

        self.path = path
        self.transform = transform
        df_list = []
        list_users = os.listdir(path)
        total_num_users = len(list_users)
        dim = []
        for idx in range(total_num_users):
            df = pd.read_csv(path + str(idx))
            df_list.append(df)
            dim.append(df.shape[0])

        self.df = pd.concat(df_list)


    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.df.loc[idx, 'labels'])
        input = np.array(ast.literal_eval(self.df.loc[idx,'pixels'])).reshape(28, 28)
        if self.transform:
            input = self.transform(input)
        sample = {'input': input, 'label': label}

        return sample




def getDataloaderList(path, transform, batch_size, shuffle):
    list_users = os.listdir(path)
    total_num_users = len(list_users)
    dl_list = []
    for idx in range(total_num_users):
        dataset = ClientDataset(path+str(idx), transform)
        dl_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))

    return dl_list
