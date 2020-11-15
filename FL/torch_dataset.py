import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import ast
import pandas as pd
from operator import itemgetter
import random
random.seed(0)

import os


class ClientDataset(Dataset):

    def __init__(self, path=None, transform=None, df=None):

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


def getDataloaderList(path, transform, batch_size, shuffle):
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
        df_list[idx] = df_list[idx].sample(frac=1, random_state=0).reset_index(drop=True)
    return df_list

def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def getClientDF(classes, df_list):

    for cl_ind, c in enumerate(classes):
        if 0 in classes:
            n = 1692
        else:
            n = 1691
        if cl_ind==0:
            data = partition(df_list[c], n)
        else:
            tmp = partition(df_list[c], n)

            for index, d in enumerate(data):
                data[index] = data[index].append(tmp[index])
                if cl_ind == len(classes)-1:
                    data[index] = data[index].sample(frac=1, random_state=0).reset_index(drop=True)


    return data


def getDataloaderNIIDList(path, total_num_users, transform, batch_size, shuffle):

    dl_list = []
    df_list = getClassesDataframes("data_test/", 3383)

    classes_1 = range(0,5)
    classes_2 = range(5, 10)

    df_1 = getClientDF(classes_1, df_list)
    df_2 = getClientDF(classes_2, df_list)

    # print(len(df_1))
    # print(type(df_1[0]))

    # random_list = random.sample(range(3383), 3383)
    # n = 0

    for i in range(len(df_1)):
        # df_1[i].to_csv("data_nonIID/" + str(random_list[n]))
        # n += 1
        dataset = ClientDataset(transform=transform, df=df_1[i])
        dl_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))

    for i in range(len(df_2)):
        # df_2[i].to_csv("data_nonIID/" + str(random_list[n]))
        # n += 1
        dataset = ClientDataset(transform=transform, df=df_2[i])
        dl_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))

    random.shuffle(dl_list)

    return dl_list
