import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, models, transforms

from src import config
from src.helper import visualize,evaluate_fn
from src.helper import run_epochs


# Define the model architecture   
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self,dataset_str):
        super(SimpleCNN, self).__init__()
        self.dataset_str = dataset_str
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=100352, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.optimizer = torch.optim.Adam(self.parameters(),lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.criterion = nn.BCELoss()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

    
    def fit(self,train_loader,val_loader):
        df,_,_ = run_epochs(train_loader,val_loader,self,self.optimizer,self.criterion,self.fc1.weight.device)
        visualize(df,self.dataset_str,"cnn")

    def predict(self,dataloader):
        _,_,y_pred,_ = evaluate_fn(dataloader, self, self.criterion)
        return y_pred


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self,dataset_str):
#         super(SimpleCNN, self).__init__()
#         self.dataset_str = dataset_str
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    #     self.fc1 = nn.Linear(in_features=32*56*56, out_features=64)
    #     self.fc2 = nn.Linear(in_features=64, out_features=1)
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #     self.criterion = nn.BCEWithLogitsLoss()
    #     self.dropout = nn.Dropout(0.5)

    # def forward(self, x):
    #     bs, c1, c2, c3 = x.size()
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.dropout(x)
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.dropout(x)
    #     x = x.view(bs, -1)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     return x.squeeze(-1)
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.fc1 = nn.Linear(in_features=128*28*28, out_features=256)
    #     self.fc2 = nn.Linear(in_features=256, out_features=128)
    #     self.fc3 = nn.Linear(in_features=128, out_features=1)
    #     self.dropout = nn.Dropout(p=0.5)
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #     self.criterion = nn.BCEWithLogitsLoss()

    # def forward(self, x):
    #     bs, c1, c2, c3 = x.size()
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.pool(F.relu(self.conv3(x)))
    #     x = x.view(bs, -1)
    #     x = self.dropout(F.relu(self.fc1(x)))
    #     x = self.dropout(F.relu(self.fc2(x)))
    #     x = self.fc3(x)
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(32)
    #     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(64)
    #     self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    #     self.bn3 = nn.BatchNorm2d(128)
    #     self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    #     self.bn4 = nn.BatchNorm2d(256)
        
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(p=0.5)
        
    #     self.fc1 = nn.Linear(256 * 56 * 56, 1024)
    #     self.fc2 = nn.Linear(1024, 1)
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #     self.criterion = nn.BCEWithLogitsLoss()
        
    # def forward(self, x):
    #     bs = x.size(0)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
        
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.relu(x)
        
    #     x = self.pool(x)
    #     x = self.dropout(x)
        
    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = self.relu(x)
        
    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #     x = self.relu(x)
        
    #     x = self.pool(x)
    #     x = self.dropout(x)
        
    #     x = x.view(bs, -1)
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
        
    #     x = self.fc2(x)
        
    #     return x.squeeze(-1)

    
    # def fit(self, train_loader, val_loader):
    #     df, _, _ = run_epochs(train_loader, val_loader, self, self.optimizer, self.criterion, self.fc1.weight.device)
    #     visualize(df, "cnn")

    # def predict(self, dataloader):
    #     _, _, y_pred, _ = evaluate_fn(dataloader, self, self.criterion)
    #     return y_pred


    
    # def fit(self,train_loader,val_loader):
    #     df,_,_ = run_epochs(train_loader,val_loader,self,self.optimizer,self.criterion,self.fc1.weight.device)
    #     visualize(df,self.dataset_str,"cnn")

    # def predict(self,dataloader):
    #     _,_,y_pred,_ = evaluate_fn(dataloader, self, self.criterion)
    #     return y_pred
