import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import ast
import pandas as pd
from FL.torch_dataset import ClientDataset


# class DatasetSplit(Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """
#
#     def __init__(self, transform, idxs):
#         self.transform = transform
#         self.idxs = [int(i) for i in idxs]
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return torch.tensor(image), torch.tensor(label)



class LocalUpdate(object):
    def __init__(self, transform, id, criterion, local_epochs):
        # self.trainloader = self.train_loader(dataset, idxs)
        self.id = id
        self.transform = transform
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.local_epochs = local_epochs

    # def train_loader(self, dataset, idxs):
    #     """
    #     Returns train, validation and test dataloaders for a given dataset
    #     and user indexes.
    #     """
    #     # split indexes for train, validation, and test (80, 10, 10)
    #     idxs_train = idxs[:int(0.8*len(idxs))]
    #
    #     trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
    #                              batch_size=8, shuffle=True)
    #     return trainloader


    #    dl = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    #    return dl

    def update_weights(self, model):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD( model.parameters(), lr=0.001, momentum=0.9)

        dataset = ClientDataset(path='data/'+str(self.id), transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        for iter in range(self.local_epochs):
            batch_loss = []
            correct = 0
            for (i, data) in enumerate(dataloader):
                images, labels = data['input'].to(self.device), data['label'].to(self.device)

                model.zero_grad()
                log_probs = model(images.double())
                loss = self.criterion(log_probs, labels)
                _, preds = torch.max(log_probs, 1)
                correct += torch.sum(preds == labels).cpu().numpy()

                loss.backward()
                optimizer.step()


                #batch_loss.append(loss.item())
            #epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), correct, len(dataset)
