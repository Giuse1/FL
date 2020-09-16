from statistics import mean
from FL.FL_user import LocalUpdate
import copy
import torch
from FL.FL_getDataset import *
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
    valset = datasets.MNIST('', download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    #mnist_noniid_dataset = get_train_dataset(trainset, num_users)

    for round in range(num_rounds):
        print('Epoch {}/{}'.format(round, num_rounds - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                local_weights = []
                local_correct = []
                local_total = []

                for idx in range(total_num_users):

                    local_model = LocalUpdate(transform=transform, id=idx, criterion=criterion, local_epochs=local_epochs)
                    w, correct, total = local_model.update_weights(
                        model=copy.deepcopy(global_model).double())
                    local_weights.append(copy.deepcopy(w))
                    #local_avg_losses.append(copy.deepcopy(loss))
                    local_correct.append(copy.deepcopy(correct))
                    local_total.append(copy.deepcopy(total))
                    #print(correct)
                    #print(total)
                    #print('{} Acc: {:.4f}'.format(phase, sum(local_correct)/sum(local_total)))




                global_weights = average_weights(local_weights)
                global_model.load_state_dict(global_weights)

                #train_loss.append(mean(local_avg_losses))
                #train_acc.append(mean(local_avg_acc))
                print('{} Acc: {:.4f}'.format(phase, sum(local_correct)/sum(local_total)))


            else:
                val_loss_r, val_accuracy_r = model_evaluation(model=global_model,dataloader=valloader, criterion=criterion)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, mean(val_loss_r), mean(val_accuracy_r)))

                val_loss.append(val_loss_r)
                val_acc.append(val_accuracy_r)


    return val_acc #train_acc, val_acc

def model_evaluation(model, dataloader, criterion):

        model.eval()  # Set model to evaluate mode
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        running_loss = 0.0
        running_corrects = 0


        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)


            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))

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
