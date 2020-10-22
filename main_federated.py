from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from model import CNNMnist, shufflenet
from FL.FL_train import train_model_aggregated
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
torch.manual_seed(0)


num_rounds = 50
local_epochs = 1
num_users = 150
users_per_group = 2
batch_size = 8
learning_rate = 0.01


print(f"NUM_USERS: {num_users}")
print(f"users_per_group: {users_per_group}")
print(f"num_rounds: {num_rounds}")
print(f"local_epochs: {local_epochs}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")

num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = shufflenet(num_classes)
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()

train_loss, train_acc, val_loss, val_acc = train_model_aggregated(model_ft, criterion, num_rounds=num_rounds, local_epochs=local_epochs, num_users=num_users,
                                                       users_per_group=users_per_group, batch_size=batch_size, learning_rate=learning_rate)

