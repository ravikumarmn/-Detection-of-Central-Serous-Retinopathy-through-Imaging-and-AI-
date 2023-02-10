# import argparse
# import json
# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
import torch
# import torch.nn as nn
# import torch.optim as optim

import config
import wandb

# from utils import (get_result, init_wandb, load_split_data, save_analysis,
#                    save_confusion, visualize)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train_fn(dataloader, model, optimizer,criterion,device = "cpu"):
    model.train()
    model.to(device)
    train_loss =list()
    total = 0
    correct = 0
    for inputs,labels in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        train_loss.append(loss.item())
        correct += (predicted == labels).sum().item()
    train_accuracy = 100 * correct / total
    training_loss =  sum(train_loss)/len(train_loss)
    return training_loss,train_accuracy
    



