from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from model import CNNMnist
from FL.FL_train import train_model, train_model_aggregated
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
torch.manual_seed(0)


num_rounds = 50
local_epochs = 1
num_users = 10
batch_size = 8
learning_rate = 0.01
model_name = "CNNMnist"


print(f"NUM_USERS: {num_users}")
print(f"num_rounds: {num_rounds}")
print(f"local_epochs: {local_epochs}")
print(f"model_name: {model_name}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")

num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model(num_classes=10, num_channels=1):

    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model_ft = CNNMnist(num_channels=num_channels, num_classes=num_classes)

    return model_ft

model_ft = initialize_model(num_classes, num_channels=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()

train_loss, train_acc, val_loss, val_acc = train_model(model_ft, criterion, num_rounds=num_rounds, local_epochs=local_epochs, num_users=num_users,
                                                       batch_size=batch_size, learning_rate=learning_rate)

val_acc = [x.cpu().numpy() for x in val_acc]

plt.plot(train_loss, label="train_loss")
plt.plot(train_acc, label="train_acc")
plt.plot(val_loss, label="vall_loss")
plt.plot(val_acc, label="val_acc")
plt.legend()
plt.legend()
plt.show()

np.savetxt(f'content/drive/My Drive/train_loss_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u_{batch_size}b_{learning_rate}lr.txt', train_loss)
np.savetxt(f'content/drive/My Drive/train_acc_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u_{batch_size}b_{learning_rate}lr.txt', train_acc)
np.savetxt(f'content/drive/My Drive/val_loss_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u_{batch_size}b_{learning_rate}lr.txt', val_loss)
np.savetxt(f'content/drive/My Drive/val_acc_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u_{batch_size}b_{learning_rate}lr.txt', np.array(val_acc))
