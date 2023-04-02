import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.metrics import confusion_matrix

from src import config
import wandb
from src.utils import save_confusion, train_fn


def visualize(dataframe,dataset_str,name = 'cnn'):

    fig = px.line(dataframe,x = "epochs",y = ['train_loss','val_loss'],
    labels={"value":"loss"},
    title="Epoch vs loss")

    fig.write_image(f"results/{dataset_str}/{name}_loss.png")

    fig = px.line(dataframe,x = "epochs",y = ['train_acc','val_acc'],
    labels={"value":"accuracy"},
    title="Epoch vs accuracy")
    
    fig.write_image(f"results/{dataset_str}/{name}_acc.png")



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

        pred_prob = model(images)
        # pred_prob = torch.sigmoid(outputs)

        predicted = torch.tensor([int(p >= 0.5) for p in pred_prob])
        # predicted = [int(p >= 0.5) for p in pred_prob]
        # _, predicted = torch.max(outputs.data, 1)
        
        # loss = criterion(outputs,labels)
        loss = criterion(pred_prob.round().float(), labels.float())
        total += labels.size(0)
        test_loss.append(loss.item())
        correct += (predicted == labels).sum().item()
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())
    testing_loss = sum(test_loss)/len(test_loss)
    test_accuracy = 100 * correct / total
    return testing_loss,test_accuracy,y_pred,y_true


def run_epochs(train_loader,val_loader,model,optimizer,criterion,device):
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    print(f"Device is set to {device}\n")
    val_loss= np.inf

    epoch_train_loss = list()
    epoch_train_acc  = list()
    epoch_val_loss   = list()
    epoch_val_acc    = list()
    ep = 0
    for epoch in range(params["EPOCHS"]):
        training_loss,train_accuracy = train_fn(train_loader, model, optimizer, criterion,device)
        validation_loss,val_accuracy,y_pred,y_true = evaluate_fn(val_loader, model, criterion)
        
        print(f"Training loss : {training_loss:.2f}\tTesting loss : {validation_loss:.2f}\tTraining Accuracy : {train_accuracy:.2f}\tTesting Accuracy : {val_accuracy:.2f}")
        if validation_loss < val_loss:
            val_loss = validation_loss
            early_stopping = 0
            torch.save(
                {
                    "model_state_dict" : model.state_dict(),
                    "params" : params
                },params['WORKING_DIR']+f"checkpoints/data_{params['DATASET_NAME'].lower()}_model_{params['MODEL_STR']}_learning_rate_{params['LEARNING_RATE']}_batch_size_{params['BATCH_SIZE']}.pt"
            )
        else:
            early_stopping += 1
        if early_stopping == params["PATIENCE"]:
            print("Early stopping, training completes")
            print(f"\nTraning_accuracy : {train_accuracy}\tTesting_accuracy : {val_accuracy}")
            print(f"Model checkpoints saved to {params['WORKING_DIR']}checkpoints/data_{params['DATASET_NAME'].lower()}_model_{params['MODEL_STR']}_learning_rate_{params['LEARNING_RATE']}_batch_size_{params['BATCH_SIZE']}.pt")
            save_confusion(y_true,y_pred, params['MODEL_STR'])
            break
        ep += 1
        epoch_train_loss.append(training_loss)
        epoch_val_loss.append(validation_loss)
        epoch_train_acc.append(train_accuracy)
        epoch_val_acc.append(val_accuracy)


        wandb.log({"epoch": epoch, "training_loss": training_loss, "val_loss": validation_loss,"training_acc":train_accuracy,"val_acc":val_accuracy})
    df = pd.DataFrame()
    df['epochs'] = list(range(ep))
    df['train_loss'] = epoch_train_loss
    df['val_loss'] = epoch_val_loss
    df['train_acc'] = epoch_train_acc
    df['val_acc'] = epoch_val_acc
    return df,y_pred,y_true