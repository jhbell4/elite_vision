import os

import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

from image_handling import imshow
from neural_network import Net

def main():
    # define directory
    dirname = os.path.dirname(__file__)
    data_directory = os.path.join(dirname, 'vision_dataset\\')
    train_directory = os.path.join(data_directory, 'train')

    # python image library of range [0, 1] 
    # transform them to tensors of normalized range[-1, 1]

    transform = transforms.Compose( # composing several transforms together
        [transforms.ToTensor(), # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((294,294))]) # mean = 0.5, std = 0.5

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform)
    classes = trainset.classes
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)#lr=0.001

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 25 == 24:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 25))
                running_loss = 0.0

    # whatever you are timing goes here
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))  # milliseconds

    # save
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()