import argparse
# import numpy as np
# import torch
# import wandb
# import pandas as pd
# from train import train_fn
# from utils import evaluate_fn,save_confusion

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import config
from models.cnn import SimpleCNN
from utils import get_result, init_wandb,save_analysis


def train_model(device,params,arg_params):
    classifier = dict(
        cnn = SimpleCNN(params['DATASET_NAME']).to(device),
        svm = svm.SVC(probability=True),
        rf = RandomForestClassifier()
    )
    model = classifier.get(arg_params.model,None)
    if model is None:
        raise ValueError
    get_result(model, arg_params.model, params["RESULT_FILE"])


def main(arg_params):
    args =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    init_wandb(args,arg_params)
    if arg_params.device:
        device = arg_params.device
    else:
        device = args['device']

    train_model(device,args,arg_params)
    




if __name__=="__main__":
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb",
        choices=["online","disabled","offline"],
        default="disabled",
        help="Enter wandb.ai mode (online or disabled",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=['cpu',"cuda"],
        help="Choice your device to train the model.",
    )
    parser.add_argument(
        "--model",
        default='cnn',
        choices=['cnn','rf','svm'],
        help= "choose which model to train."
    )

    args = parser.parse_args()


    main(args)
    save_analysis(params["RESULT_FILE"])

