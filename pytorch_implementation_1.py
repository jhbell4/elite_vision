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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((294,294))]) # mean = 0.5, std = 0.5

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_directory, transform=transform)

    classes = trainset.classes

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # imshow(torchvision.utils.make_grid(images))

    # # print the class of the image
    # print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

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

class Net(nn.Module):

    def __init__(self):
        ''' initialize the network '''
        super(Net, self).__init__()
        self.dim = 294
        self.kernel_size = 9
        self.kernel_stride = 3
        self.pooling_size = 2
        self.number_of_classes = 14

        self.dim_conv1 = (self.dim - self.kernel_size)/self.kernel_stride + 1
        self.dim_pool1 = self.dim_conv1/self.pooling_size
        self.dim_conv2 = (self.dim_pool1 - self.kernel_size)/self.kernel_stride + 1
        self.dim_pool2 = int(self.dim_conv2/self.pooling_size)
    # 3 input image channel, 6 output channels, 
    # kxk square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, self.kernel_size,stride=self.kernel_stride)
    # Max pooling over a (p,p) window
        self.pool = nn.MaxPool2d(self.pooling_size, self.pooling_size)
        self.conv2 = nn.Conv2d(6, 16, self.kernel_size,stride=self.kernel_stride) 
        
        self.fc1 = nn.Linear(16 * self.dim_pool2 * self.dim_pool2, 120)# 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.number_of_classes)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.dim_pool2 * self.dim_pool2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    main()

