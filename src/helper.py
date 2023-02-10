# import config
import numpy as np
import wandb
import torch
import pandas as pd
import os
import plotly.express as px

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def visualize(dataframe,name = 'cnn'):
    if not os.path.exists("results"):
        os.mkdir("results")
    
    fig = px.line(dataframe,x = "epochs",y = ['train_loss','val_loss'],
    labels={"value":"loss"},
    title="Epoch vs loss")

    fig.write_image(f"results/{name}_loss.png")
    fig = px.line(dataframe,x = "epochs",y = ['train_acc','val_acc'],
    labels={"value":"accuracy"},
    title="Epoch vs accuracy")
    
    fig.write_image(f"results/{name}_acc.png")



@torch.no_grad()
def evaluate_fn(dataloader, model,criterion, device = 'cpu'):
    model.eval()
    model.to(device)
    test_loss = list()
    total = 0
    correct = 0
    y_pred = list()
    y_true = list()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.long().to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs,labels)
        total += labels.size(0)
        test_loss.append(loss.item())
        correct += (predicted == labels).sum().item()
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())
    testing_loss = sum(test_loss)/len(test_loss)
    test_accuracy = 100 * correct / total
    return testing_loss,test_accuracy,y_pred,y_true