#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

import json

import argparse

from model_setup import load_model, predict
from processing import load_data, process_image

# Setup a parser.

def arg_parser():
   
    parser = argparse.ArgumentParser(description="Neural Network user input settings") 

   #Add variables that give user the option to change 
    parser.add_argument('image_path', 
                        #default='/ImageClassifier/flowers/test/1/image_06752.jpg', 
                        type =str,
                        help='Filepath of image, expects a jpeg')
    parser.add_argument('--arch', 
                        default= 'vgg16', 
                        type = str,
                        help='Choose between vgg16 or resnet' )
    parser.add_argument('--checkpoint', 
                        default='./checkpoint.pth', 
                        type = str,
                        help='Filepath of the saved model')
    parser.add_argument('--top_k', 
                        default=5, 
                        type=int,
                        help='How many flowers and probs you want it to print out. Integer')
    parser.add_argument('--gpu', 
                        default="gpu", 
                        action="store", 
                        help='Option to swap between gpu/cpu')
    parser.add_argument('--category_names', 
                        default='cat_to_name', 
                        type = str,
                        help='Filepath of the saved model')
    
    # Parse args
    args = parser.parse_args()
    
    return args

args=arg_parser()
print(args)

image_path=args.image_path
arch=args.arch
filepath=args.checkpoint
top_k=args.top_k
gpu=args.gpu
category_names =args.category_names

# Load model

model_load=load_model(filepath,gpu,arch)

print(model_load)
 
print("Model load complete") 


#Label mapping

cat_names_filepath='./ImageClassifier/' + category_names  + '.json'
print(cat_names_filepath)


with open(cat_names_filepath, 'r') as f:
    cat_to_name  = json.load(f)


# Predict most like k flowers the image is

prob_arr, pred_indexes, pred_classes, pred_names=predict(image_path, model_load,cat_to_name,top_k)  



print(prob_arr, pred_indexes, pred_classes, pred_names)

# Print name of predicted flower with highest probability
print(f"This is most likely to be a: '{pred_names[0]}', with a probability of {round(prob_arr[0]*100,2)}% ")









#python ImageClassifier/predict.py ImageClassifier/flowers/test/1/image_06743.jpg
#python ImageClassifier/predict.py ImageClassifier/flowers/test/1/image_06743.jpg --arch resnet18
