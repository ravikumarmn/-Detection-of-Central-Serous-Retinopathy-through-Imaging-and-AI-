import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, models, transforms

from src import config
from src.helper import visualize,evaluate_fn
from src.utils import run_epochs


# Define the model architecture   
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=100352, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
        self.optimizer = torch.optim.Adam(self.parameters(),lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bs,_,_,_ = x.size()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(bs,-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fit(self,train_loader,val_loader):
        df,_,_ = run_epochs(train_loader,val_loader,self,self.optimizer,self.criterion,self.fc1.weight.device)
        visualize(df,"cnn")

    def predict(self,dataloader):
        _,_,y_pred,_ = evaluate_fn(dataloader, self, self.criterion)
        return y_pred


