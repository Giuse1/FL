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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
torch.manual_seed(0)


batch_size = 8
num_rounds = 2
local_epochs = 1
num_users = 150

model_name = "resnet"

num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model(num_classes=10, num_channels=1):

    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # input_size = 28
    else:
        model_ft = CNNMnist(num_channels=num_channels, num_classes=num_classes)

    return model_ft

model_ft = initialize_model(num_classes, num_channels=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ft = model_ft.to(device)

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
#trainset = datasets.MNIST('', download=True, train=True, transform=transform)
#valset = datasets.MNIST('', download=True, train=False, transform=transform)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
#dataloaders_dict = {'train': trainloader, 'val': valloader }

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_loss, train_acc, val_loss, val_acc = train_model(model_ft, criterion, num_rounds=num_rounds, local_epochs=local_epochs, num_users=num_users)
val_acc = [x.cpu().numpy() for x in val_acc]

plt.plot(train_loss, label="train_loss")
plt.plot(train_acc, label="train_acc")
plt.plot(val_loss, label="vall_loss")
plt.plot(val_acc, label="val_acc")
plt.legend()
plt.legend()
plt.show()

np.savetxt(f'train_loss_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u.txt', train_loss)
np.savetxt(f'train_acc_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u.txt', train_acc)
np.savetxt(f'val_loss_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u.txt', val_loss)
np.savetxt(f'val_acc_{model_name}_f_{num_rounds}r_{local_epochs}le_{num_users}u.txt', np.array(val_acc))
