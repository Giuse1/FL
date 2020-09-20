from statistics import mean
from FL.FL_user import LocalUpdate
import copy
import torch
from FL.FL_getDataset import *
from FL.torch_dataset import getDataloaderList
from torch.utils.data import DataLoader
import os


def train_model(global_model, criterion, num_rounds=50, local_epochs=1):
    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    list_users = os.listdir('data')
    total_num_users = len(list_users)
    num_users = 10

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

    # valset = ValidationDataset(path='data_test/', transform=transform)
    # valloader = DataLoader(valset, batch_size=8, shuffle=True)
    trainloader_list = getDataloaderList(path='data/', transform=transform, batch_size=8, shuffle=True)
    valloader_list = getDataloaderList(path='data_test/', transform=transform, batch_size=8, shuffle=True)
    # mnist_noniid_dataset = get_train_dataset(trainset, num_users)

    for round in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                gloabl_num_correct = 0
                global_num_total = 0
                global_loss = 0

                for idx in range(total_num_users):
                    local_model = LocalUpdate(dataloader=trainloader_list[idx], transform=transform, id=idx, criterion=criterion,
                                               local_epochs=local_epochs)
                    w, local_loss, local_correct, local_total = local_model.update_weights(
                        model=copy.deepcopy(global_model).double())
                    local_weights.append(copy.deepcopy(w))
                    gloabl_num_correct += local_correct
                    global_num_total += local_total
                    global_loss += local_loss
                    # local_avg_losses.append(copy.deepcopy(loss))

                    # print(correct)
                    # print(total)
                    # print('{} Acc: {:.4f}'.format(phase, sum(local_correct)/sum(local_total)))

                global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)

                # train_loss.append(mean(local_avg_losses))
                # train_acc.append(mean(local_avg_acc))
                train_loss.append(global_loss / global_num_total)
                train_acc.append(gloabl_num_correct / global_num_total)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, global_loss/global_num_total, gloabl_num_correct/global_num_total ))


            else:
                val_loss_r, val_accuracy_r = model_evaluation(model=global_model.double(),
                                                              dataloader_list=valloader_list, criterion=criterion)

                val_loss.append(val_loss_r)
                val_acc.append(val_accuracy_r)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss_r, val_accuracy_r))


    return train_loss, train_acc, val_loss, val_acc


def model_evaluation(model, dataloader_list, criterion):
    model.eval()  # Set model to evaluate mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    for dataloader in dataloader_list:

        for (i, data) in enumerate(dataloader):
            inputs, labels = data['input'].to(device), data['label'].to(device)

            outputs = model(inputs.double())
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            running_total += labels.shape[0]

    epoch_loss = running_loss / running_total
    epoch_acc = running_corrects.double() / running_total


    return epoch_loss, epoch_acc


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
