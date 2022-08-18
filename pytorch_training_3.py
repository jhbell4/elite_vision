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
from torch.optim import lr_scheduler

from image_handling import imshow
from neural_network import preprocess_transform
from neural_network import validation_transform

import csv

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # define directory
    dirname = os.path.dirname(__file__)
    data_directory = os.path.join(dirname, 'vision_dataset_combined\\')
    train_directory = os.path.join(data_directory, 'train')
    val_directory = os.path.join(data_directory, 'val')

    # python image library of range [0, 1] 
    # transform them to tensors of normalized range[-1, 1]

    train_transform = preprocess_transform()
    valid_transform = validation_transform()

    # set batch_size
    batch_size = 400

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=train_transform)
    validset = torchvision.datasets.ImageFolder(root=val_directory, transform=valid_transform)
    dataset_sizes = {'train': len(trainset), 'val': len(validset)}

    class_names = trainset.classes
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders = {'train': trainloader, 'val': validloader}
    
    # Model Building
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    print(model_ft)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#lr=0.001, momentum=0.9
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)#lr=0.001
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    csv_file = open('accuracy.csv', 'w')
    csv_writer = csv.writer(csv_file)

    start.record()

    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, csv_writer, num_epochs=100)

    # whatever you are timing goes here
    end.record()

    csv_file.close()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))  # milliseconds

    # save
    PATH = './cifar_net.pth'
    torch.save(model.state, PATH)

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, csvwriter, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        csv_data = [epoch]
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            csv_data.extend([epoch_loss,epoch_acc])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        csvwriter.writerow(csv_data)
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    main()