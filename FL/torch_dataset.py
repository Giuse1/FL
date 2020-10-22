import torch
import torchvision
import torchvision.transforms as transforms
from operator import itemgetter
import random
random.seed(0)


def get_cifar_iid(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_dataset = []
    test_dataset = []
    sets = [trainset, testset]
    for idx, s in enumerate(sets):
        for d in s:
            d = list(d)
            d[1] = classes[d[1]]
            if idx == 0:
                train_dataset.append(tuple(d))
            else:
                test_dataset.append(tuple(d))

    final_classes = ('plane', 'car', 'ship', 'truck', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse')

    trainset = []
    testset = []
    sets = [train_dataset, test_dataset]
    for idx, s in enumerate(sets):
        for d in s:
            d = list(d)
            d[1] = final_classes.index(d[1])
            if idx == 0:
                trainset.append(tuple(d))
            else:
                testset.append(tuple(d))

    data_per_client = 100
    total_data = len(trainset)
    random_list = random.sample(range(total_data), total_data)
    num_clients = int(total_data / data_per_client)
    datasets = []
    for i in range(num_clients):
        indexes = random_list[i:i + 100]
        datasets.append(itemgetter(*indexes)(trainset))

    trainloader_list = []
    for d in datasets:
        trainloader_list.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader_list, testloader


# def get_cifar_noniid(batch_size):
#     indexes_animals
#     indexes_means
#
#

