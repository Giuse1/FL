from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

from model import CNNMnist
from FL.FL_train import train_model

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
torch.manual_seed(0)

FILE_NAME = 'acc_custom_f_50e.txt'
BATCH_SIZE = 8
NUM_ROUNDS = 50
LOCAL_EPOCHS = 1
NUM_CLASSES = 10
MODEL_NAME = "CNNmnist"


def initialize_model(model_name, num_classes=10, num_channels=1):
    if model_name == "resnet":
        model= models.resnet18(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 28

    else:
        model = CNNMnist(num_channels=num_channels, num_classes=num_classes)

    return model


model_ft = initialize_model(MODEL_NAME, NUM_CLASSES, num_channels=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ft = model_ft.to(device)

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

hist_acc = train_model(model_ft, criterion, num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS)

hist_acc = [x.cpu().numpy() for x in hist]
plt.plot(hist_acc)
plt.show()
hist_acc_arr = np.array(hist_acc)
np.savetxt(FILE_NAME, hist_acc_arr)
