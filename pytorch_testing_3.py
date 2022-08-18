import os
import time
import copy

import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
#import torchvision.transforms as transforms # transform data
from torchvision import datasets, models, transforms
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

from image_handling import imshow
from neural_network import validation_transform

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # # Define Test Set Parameters
    # set batch_size
    batch_size = 4
    # set number of workers
    num_workers = 2

    transform = validation_transform()

    dirname = os.path.dirname(__file__)
    data_directory = os.path.join(dirname, 'vision_dataset_combined\\')
    test_directory = os.path.join(data_directory, 'test')
    testset = torchvision.datasets.ImageFolder(root=test_directory, transform=transform)
    class_names = testset.classes
    print(class_names)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ## Load Network

    # Model Building
    #model_ft = models.resnet18(pretrained=True)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    PATH = './cifar_net.pth'
    #model_ft.load_state_dict(torch.load(PATH))
    model_ft = torch.load(PATH)
    model_ft = model_ft.to(device)

    # Check Accuracy
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model_ft(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

if __name__ == '__main__':
    main()
