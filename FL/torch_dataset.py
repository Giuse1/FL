from torch.utils.data import Dataset
import torch
import numpy as np
import ast
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

import os


class ClientDataset(Dataset):

    def __init__(self, path, transform=None, df=None):

        self.path = path
        self.transform = transform
        if df is not None:
            self.df = df
        else:
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


def getDataloaderList(path, total_num_users, transform, batch_size, shuffle):
    list_users = os.listdir(path)
    total_num_users = len(list_users)
    dl_list = []
    for idx in range(total_num_users):
        dataset = ClientDataset(path+str(idx), transform)
        dl_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))

    return dl_list

def getClassesDataframes(path, total_num_users):


    df_list = [pd.DataFrame() for _ in range(10)]

    for idx in range(total_num_users):
        df = pd.read_csv(path + str(idx))[['labels','pixels']]
        for label in range(10):
            df_list[label] = df_list[label].append(df[df['labels']==label])

    for idx in range(10):
        df_list[idx] = df_list[idx].sample(frac=1).reset_index(drop=True)
    return df_list



def getClientDF(classes, last, df_list, lenghts_data):

    df = pd.DataFrame()

    for c in classes:
        if last:
            to_append = df_list[c]
        else:
            to_append = df_list[c].sample(n=int(lenghts_data[c]/75))


        df = df.append(to_append)
        df_list[c] = df_list[c].drop(to_append.index)


    return df.sample(frac=1).reset_index(drop=True)


def getDataloaderNIIDList(path, total_num_users, transform, batch_size, shuffle):

    dl_list = []
    df_list = getClassesDataframes(path, total_num_users)
    lenghts_data = [len(c) for c in df_list]

    for idx in range(total_num_users):

        if idx%2 == 0:
            classes = range(0,5)
        else:
            classes = range(5, 10)

        if idx==total_num_users-1 or idx==total_num_users-2:
            last=True
        else:
            last= False

        df = getClientDF(classes, last, df_list, lenghts_data)

        dataset = ClientDataset(path + str(idx), transform, df)
        dl_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))


    return dl_list