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

def main():
    # define directory
    dirname = os.path.dirname(__file__)
    data_directory = os.path.join(dirname, 'vision_dataset\\')
    train_directory = os.path.join(data_directory, 'train')
    test_directory = os.path.join(data_directory, 'test')

    # python image library of range [0, 1] 
    # transform them to tensors of normalized range[-1, 1]

    transform = transforms.Compose( # composing several transforms together
        [transforms.ToTensor(), # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_directory, transform=transform)

    classes = trainset.classes
    print(classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))

    # print the class of the image
    print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

class Net(nn.Module):

    def __init__(self):
        ''' initialize the network '''
        super(Net, self).__init__()
    # 3 input image channel, 6 output channels, 
    # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
    # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)# 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)




if __name__ == '__main__':
    main()

