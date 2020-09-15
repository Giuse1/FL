from torch.utils.data import Dataset
import torch
import numpy as np
import ast
import pandas as pd
from torchvision import transforms


class ClientDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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
