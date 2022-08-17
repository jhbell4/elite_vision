import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

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