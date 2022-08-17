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
    # # Define Test Set Parameters
    # set batch_size
    batch_size = 4
    # set number of workers
    num_workers = 2

    transform = transforms.Compose( # composing several transforms together
        [transforms.ToTensor(), # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((294,294))]) # mean = 0.5, std = 0.5

    dirname = os.path.dirname(__file__)
    data_directory = os.path.join(dirname, 'vision_dataset\\')
    test_directory = os.path.join(data_directory, 'test')
    testset = torchvision.datasets.ImageFolder(root=test_directory, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ## Load Network

    net = Net()
    PATH = './cifar_net.pth'
    net.load_state_dict(torch.load(PATH))

    # Check Accuracy
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

if __name__ == '__main__':
    main()
