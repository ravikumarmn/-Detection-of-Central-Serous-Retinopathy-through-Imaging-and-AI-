import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
# from sklearn.model_selection import train_test_split
# from train import train_fn
import wandb
from src import config


def save_confusion(y_true,y_pred,name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(f'results/{config.DATASET_NAME}/{name}_confustion.png')
    # plt.show()
def load_split_data(file_name_str = "train_val_test",model_type = None):
    datasets = torch.load(f"dataset/{config.DATASET_NAME}_{file_name_str}.pt")

    # Split the data into training and validation sets
    train_data = datasets["train"]['train_images'] # The training data tensor
    train_labels = datasets["train"]['train_labels'] # The training labels tensor

    val_data = datasets["validation"]['val_images'] # The validation data tensor
    val_labels = datasets["validation"]['val_labeld'] # The validation labels tensor

    # test_data = datasets['test']['test_images']
    # test_labels = datasets['test']['test_labels']

   
    if model_type == "cnn":
        print(f"Traning data size : {datasets['metadata']['train_size']}")
        train_dataset = torch.utils.data.TensorDataset(train_data,train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_data,val_labels)
        # test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
        return train_loader,val_loader

    elif model_type == "svm" or "random forest":
        T, V = len(train_data), len(val_labels)
        return train_data.view(T,-1).numpy(), val_data.view(V, -1).numpy(), train_labels.numpy(), val_labels.numpy()
    else:
        raise NotImplementedError



def init_wandb(params,arg_params):
    wandb.init(
        config=params,
        project="detect-central-cerous-retinopathy",
        entity="ravikumarmn",
        name=f'{params["MODEL_STR"]}_batch_size_{params["BATCH_SIZE"]}_learning_rate_{params["LEARNING_RATE"]}',
        group="binary classification",
        notes = f"detecting central cerous retinopathy using model {params['MODEL_STR']}.",
        tags=[params['MODEL_STR']],
        mode=arg_params.wandb)


def save_analysis(json_file_str):
    result = json.load(open(json_file_str,"r"))

    values = list()
    for val in result.values():
        values.append(list(val.values()))
    vals = [list(val.keys())] + values
    model_name= [" "] + list(result.keys())
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=model_name,
        fill_color='grey',
        align=['left','center'],
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=vals,
        
        line_color='darkslategray',
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 11)
        ))
    ])
    fig.write_image(f"results/{config.DATASET_NAME}/performance_analysis.png")

def edit_json(file,data_dict):
    json_file = json.load(open(file,"r")) # a
    json_file.update(data_dict) # update
    json.dump(json_file, open(file,"w")) # 

def get_result(model_obj,name, save_file):
    
    if name =="cnn":
        X_train, X_test = load_split_data(model_type='cnn')
        y_train = X_test
        y_test = torch.concat([y for _,y in X_test]).numpy()
    else:
        X_train, X_test, y_train, y_test = load_split_data(model_type = name)

    model_obj.fit(X_train, y_train)
    y_pred = model_obj.predict(X_test)

    analysis = dict(   
        accuracy = round(accuracy_score(y_test, y_pred),3),
        precision = round(precision_score(y_test, y_pred),3) ,
        recall = round(recall_score(y_test, y_pred),3),
        f_score = round(f1_score(y_test, y_pred),3)
        )
    if name == "svm" or "rf":
        save_model(model_obj,name)
    save_confusion(y_test, y_pred, name)
    result = {name.upper() : analysis}
    edit_json(save_file,result)
    print(f"\nAccuracy of model {name} : {analysis['accuracy']:.3f}")


def save_model(model,name):
    # saving the model 
    import pickle 
    pickle_out = open(f"checkpoints/{name}_classifier.pkl", mode = "wb") 
    pickle.dump(model, pickle_out) 
    pickle_out.close()

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

