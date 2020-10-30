#importing necessary libraries

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

from processing import load_data, process_image

from model_setup import classifier_setup, validation, train_model,test_model,save_model,load_model


# Setup a parser. Note that converts user input from the command line to arguments in the .py file.

def arg_parser():
   
    parser = argparse.ArgumentParser(description="Neural Network user input settings") 

   #Add variables that give the user the option to change 
    parser.add_argument('--data_dir', 
                        default='./ImageClassifier/flowers', 
                        type =str,
                        help='Directory of development data')
    parser.add_argument('--arch', 
                        default= 'vgg16', 
                        type = str,
                        help='Choose between vgg16 or resnet' )
    parser.add_argument('--checkpoint', 
                        default='./checkpoint.pth', 
                        type = str,
                        help='Filepath of the saved model')
    parser.add_argument('--hidden_units', 
                        default=1024,            
                        type=int,
                        help='How many nodes in the hidden layer. If using ResNet18, choose this number to be less than 512')
    parser.add_argument('--epochs', 
                        default=2,  
                        type=int,
                        help='How many epochs training is run over. Positive integer')
    parser.add_argument('--learning_rate', 
                        default=0.001, 
                        dest="learning_rate", 
                        type=float,
                        help='Learning rate of training.')
    parser.add_argument('--dropout', 
                        default=0.25,  
                        type=float,
                        help='Proportion of nodes dropped out in training.')
    parser.add_argument('--gpu', 
                        default="gpu", 
                        type=str,
                        help='Option to swap between gpu/cpu')
    
    # Parse args
    args = parser.parse_args()
    
    return args

args=arg_parser()
print(args)

# Assigning args
data_dir=args.data_dir
arch=args.arch
filepath=args.checkpoint
hidden_units=args.hidden_units
epochs=args.epochs
learning_rate=args.learning_rate
dropout=args.dropout
gpu=args.gpu


# Load and preprocess data 
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

print("Data loading complete")  


# Setting number of nodes in final layer
output_units = len(train_data.class_to_idx)

# Label mapping
with open('./ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)




# Model set up

# Load pretrained model
if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
else: 
    print("Arch choice error. Please choose either 'vgg16' or 'resnet18'")

#print(model)


# Warnings around hidden layer nodes number
if arch == 'vgg16' and hidden_units>25088:
     print("Hidden_layer error. Please choose hidden_units as an integer less than 25088 ")
        
elif arch == 'resnet18' and hidden_units>512:
     print("Hidden_layer error. Please choose hidden_units as an integer less than 512 ")
elif hidden_units< output_units:
     print("Hidden_layer error. Please choose hidden_units as an integer less than output_units (default=102) ")
else: 
    pass


model, criterion ,optimizer =classifier_setup(model, arch, hidden_units,output_units, dropout,learning_rate)

print("Model setup complete")  


# Train model

model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer,gpu)

print("Model training complete") 



# Test model

test_model(model, testloader,gpu)

print("Model testing complete") 


# Save model

save_model(model,train_data,arch, optimizer, epochs,learning_rate,filepath)

print("Model save complete") 


