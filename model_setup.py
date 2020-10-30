import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image

import argparse
import json

from processing import process_image

 
    
def classifier_setup(model,arch, hidden_layer, output_units, dropout,learning_rate):
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

        # Load pretrained model
    if arch == 'vgg16':
        input_units = model.classifier[0].in_features
        
        classifier = nn.Sequential(OrderedDict([
                          ('inputs', nn.Linear(input_units, hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('do1', nn.Dropout(dropout)),
                          ('hidden_layer1', nn.Linear(hidden_layer, output_units)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    
    elif arch == 'resnet18':
    
        input_units = model.fc.in_features    
            
        classifier =  nn.Sequential(OrderedDict([
                          ('inputs', nn.Linear(input_units, hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('do1', nn.Dropout(dropout)),
                          ('hidden_layer1', nn.Linear(hidden_layer, output_units)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.fc = classifier
   
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    
    return model, criterion ,optimizer
       
   
       

def validation(model,criterion,validloader,gpu):
    valid_loss = 0
    accuracy = 0
    if gpu == 'gpu':
        model.to('cuda')
    else: model.cpu()
        
    for ii, (images, labels) in enumerate(validloader):
    
        if gpu == 'gpu':
            images, labels = images.to('cuda'), labels.to('cuda')
        else: 
            images, labels = images.to('cpu'), labels.to('cpu')
    
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy




def train_model(model, epochs,trainloader, validloader, criterion, optimizer,gpu):
    print_every=10
    steps = 0
    running_loss = 0
    
    if gpu == 'gpu':
        model.to('cuda')
    else: 
        model.to('cpu')

    print("Training is starting....... ")
    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else: 
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
    
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
            
            if steps % print_every == 0:
            # Eval mode for inference            
                model.eval()
            
            # Turn off gradients for validation
                with torch.no_grad():
                    valid_loss, accuracy = validation(model,criterion,validloader,gpu)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
                running_loss = 0
            
            # Make sure training is back on
                model.train()
    return model, optimizer




def test_model(model, testloader,gpu):
    correct = 0
    total = 0
    
    if gpu == 'gpu':
        model.to('cuda')
    else: 
        model.to('cpu')
        
    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            if gpu == 'gpu':
                 images, labels = images.to('cuda'), labels.to('cuda')
            else: 
                images, labels = images.to('cpu'), labels.to('cpu')
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('Accuracy of the network on the test images: {:.3f}'.format(100 * correct / total))


def save_model(model,train_data,arch, optimizer, epochs,learning_rate,filepath):
    
    model.class_to_idx = train_data.class_to_idx

    if arch == 'vgg16':
        
        checkpoint = {'input_size': model.classifier[0].in_features,
              'output_size': len(train_data.class_to_idx),
              'arch': arch,
              'learning_rate': learning_rate,
              'final_layer' : model.classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    
    elif arch == 'resnet18':
        
        checkpoint = {'input_size': model.fc[0].in_features,
              'output_size': len(train_data.class_to_idx),
              'arch': arch,
              'learning_rate': learning_rate,
              'final_layer' : model.fc,
              'epochs': epochs,
              'optimizer': optimizer.state_dict,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    return torch.save(checkpoint, filepath)
    
    

def load_model(filename,gpu,arch):
    
    if gpu == 'gpu':
        checkpoint = torch.load(filename)
    else: 
        checkpoint = torch.load(filename, map_location=lambda storage, location: storage)

    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    if arch == 'vgg16':
        model.classifier = checkpoint['final_layer']
    elif arch == 'resnet18':
        model.fc = checkpoint['final_layer']
       
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model


def predict(image_path, model, cat_to_name,top_k):
    
    # Preprocess image
    image_torch = process_image(image_path)

    # Use model to predict
    model.eval()
    
    model.to('cpu')    
    
    with torch.no_grad():
        # Running image through network
        ouput = model.forward(image_torch)
  
    #calculate probability
    ps = torch.exp(ouput)
    
    # get the top 5 results
    probs, top_idxes = ps.topk(top_k)

    prob_arr = probs.data.numpy()[0]

    pred_indexes = top_idxes.data.numpy()[0].tolist()  
    
    #Call up class to index dict
    class_to_idx = model.class_to_idx

    # Inverting class to index dict
    idx_to_class = {x: y for y, x in class_to_idx.items()}

    
    pred_classes = [idx_to_class[x] for x in pred_indexes]
    
    pred_names = [cat_to_name[str(x)] for x in pred_classes]

    print(pred_names,prob_arr)
    return prob_arr, pred_indexes, pred_classes, pred_names

