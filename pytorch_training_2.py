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

import csv

def main():
    # define directory
    dirname = os.path.dirname(__file__)
    data_directory = os.path.join(dirname, 'vision_dataset\\')
    train_directory = os.path.join(data_directory, 'train')

    # python image library of range [0, 1] 
    # transform them to tensors of normalized range[-1, 1]

    transform = transforms.Compose( # composing several transforms together
        [transforms.ToTensor(), # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # mean = 0.5, std = 0.5
        transforms.CenterCrop((281,281))]) 

    # set batch_size
    batch_size = 16

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform)
    classes = trainset.classes

    indices = torch.randperm(len(trainset)).tolist()
    dataset_train = torch.utils.data.Subset(trainset, indices[:-50])
    dataset_validate = torch.utils.data.Subset(trainset, indices[-50:])

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#lr=0.001, momentum=0.9
    optimizer = optim.Adam(net.parameters(), lr=0.001)#lr=0.001

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    csv_file = open('accuracy.csv', 'w')
    csv_writer = csv.writer(csv_file)

    start.record()

    for epoch in range(20):  # loop over the dataset multiple times
        
        # Train
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
        print('%d loss: %.3f' %(epoch + 1, running_loss))
        # Test Training Accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        training_accuracy = float(correct) / float(total)
        print(f'Training Accuracy: {100 * correct // total} %')

        # Valid Training Accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        validation_accuracy = float(correct) / float(total)
        print(f'Validation Accuracy: {100 * correct // total} %')
        csv_data = [epoch,running_loss,training_accuracy,validation_accuracy]
        csv_writer.writerow(csv_data)
        running_loss = 0


    # whatever you are timing goes here
    end.record()

    csv_file.close()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))  # milliseconds

    # save
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()