import os

import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

# define directory
dirname = os.path.dirname(__file__)
data_directory = os.path.join(dirname, 'vision_dataset/vision_dataset')
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