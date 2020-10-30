
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

import argparse
import json



def load_data(data_dir):
#Train transforms: want to augment the training set via cropping, rotating, etc.

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

#Test transforms (for both validation and test sets): just want to resize then crop to 224x224

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir , transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir , transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir , transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, testloader, validloader, train_data, test_data, valid_data



def process_image(image_path):
 # TODO: Process a PIL image for use in a PyTorch model
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image_path)

    pil_image = preprocess(pil_image).float()

    np_image = np.array(pil_image)    

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    

    np_image = np.transpose(np_image, (2, 0, 1))
  
    image_torch = torch.FloatTensor(np_image)
    image_torch = image_torch.unsqueeze(0)
    
    return image_torch


 
    
   