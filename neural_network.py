import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torchvision import datasets, models, transforms
def preprocess_transform():
    trans1 = transforms.ToTensor() # to tensor object
    trans2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # mean = 0.5, std = 0.5
    trans3 = transforms.CenterCrop((1000,1000))
    trans4 = transforms.RandomCrop((281,281))
    trans5 = transforms.RandomHorizontalFlip()
    trans6 = transforms.RandomVerticalFlip()
    transform = transforms.Compose([trans1,trans2,trans3,trans4,trans5,trans6])
    return transform

def validation_transform():
    trans1 = transforms.ToTensor() # to tensor object
    trans2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # mean = 0.5, std = 0.5
    trans3 = transforms.CenterCrop((281,281))
    transform = transforms.Compose([trans1,trans2,trans3])
    return transform

class Net(nn.Module):

    def __init__(self):
        ''' initialize the network '''
        super(Net, self).__init__()
        self.dim = 294
        self.kernel_size_1 = 7
        self.kernel_size_2 = 7
        self.kernel_size_3 = 5
        self.kernel_stride_1 = 2
        self.kernel_stride_2 = 2
        self.kernel_stride_3 = 1
        self.pooling_size = 2
        self.number_of_classes = 14

        self.dim_conv1 = (self.dim - self.kernel_size_1)/self.kernel_stride_1 + 1
        self.dim_pool1 = self.dim_conv1/self.pooling_size
        self.dim_conv2 = (self.dim_pool1 - self.kernel_size_2)/self.kernel_stride_2 + 1
        self.dim_pool2 = self.dim_conv2/self.pooling_size
        self.dim_conv3 = (self.dim_pool2 - self.kernel_size_3)/self.kernel_stride_3 + 1
        self.dim_pool3 = int(self.dim_conv3/self.pooling_size)
    # 3 input image channel, 6 output channels, 
    # kxk square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, self.kernel_size_1,stride=self.kernel_stride_1)
        self.conv2 = nn.Conv2d(16, 32, self.kernel_size_2,stride=self.kernel_stride_2) 
        self.conv3 = nn.Conv2d(32, 64, self.kernel_size_3,stride=self.kernel_stride_3)
    # Max pooling over a (p,p) window
        self.pool = nn.MaxPool2d(self.pooling_size, self.pooling_size)
    # Nonlinear transform
        self.nl_trans = F.relu

        self.fc1 = nn.Linear(64 * self.dim_pool3 * self.dim_pool3, 120)# 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.number_of_classes)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(self.nl_trans(self.conv1(x)))
        x = self.pool(self.nl_trans(self.conv2(x)))
        x = self.pool(self.nl_trans(self.conv3(x)))
        x = x.view(-1, 64 * self.dim_pool3 * self.dim_pool3)
        x = self.nl_trans(self.fc1(x))
        x = self.nl_trans(self.fc2(x))
        x = self.fc3(x)
        return x