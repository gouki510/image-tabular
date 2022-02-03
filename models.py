
   
from PIL.Image import MODES
from numpy import float16, float32
import torch
import torch.nn as nn
from torchvision import models
import random

class CNN(nn.Module):

    def __init__(self,hidden_size):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.resnet = models.resnet50(pretrained=True).cuda()
        self.resnet.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=self.hidden_size, bias=True),
                nn.ReLU(),
            )
        self.meta_fc = nn.Sequential(
            nn.Linear(in_features=4,out_features=4,bias=True),
            nn.ReLU(),
        )

    def forward(self,x,metadata):
        batch_size, frames, c, h, w = x.shape
        t = random.randint(0,5)
        x = x[:,t,:,:,:].reshape(batch_size,c,h,w)
        metadata = metadata[:,t,:].reshape(batch_size,4)
        x = self.resnet(x)
        metadata = self.meta_fc(metadata.to(torch.float32))
        x = x = torch.cat((x,metadata),-1).to(torch.float32)
        return x

class CNNLSTM(nn.Module):
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.resnet = models.resnet50(pretrained=True).cuda()
        self.resnet.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=self.hidden_size, bias=True),
                nn.ReLU(),
            )
        self.dr = nn.Dropout()
        self.lstm = nn.LSTM(
            input_size=self.hidden_size+4,  # 入力size
            hidden_size=self.hidden_size+4,  # 出力size
            num_layers=self.num_layers,  # stackする数
            dropout=0.5,
            batch_first=True,  # given_data.shape = (batch , frames , input_size)
            bidirectional=False,
        )
        self.meta_fc = nn.Sequential(
            nn.Linear(in_features=4,out_features=4,bias=True),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, metadata) -> torch.Tensor:
        assert len(x.shape) == 5, print("data shape is incorrect.")
        batch_size, frames, c, h, w = x.shape
        # shape -> (batch*frame , ch , 224 , 224)
        emb_series = []
        for t in range(frames):
            image = x[:,t,:,:,:].reshape(batch_size,c,h,w)
            emb = self.resnet(image)
            emb = self.dr(emb)
            emb_series.append(emb)
        emb_series = torch.stack(emb_series).reshape(batch_size,frames,self.hidden_size)
        metadata = self.meta_fc(metadata.to(torch.float32))
        x = torch.cat((emb_series,metadata),-1).to(torch.float32)
        self.reset_state(batch_size)
        x, (h0, c0) = self.lstm(x, (self.h0, self.c0))
        return x[:,-1,:]

    def reset_state(self, batch_size: int):
        self.h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size+4).cuda()
        self.c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size+4).cuda()
    
class DCNN(nn.Module):
    def __init__(self,
    hidden_size
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dcnn = nn.Conv3d(in_channels=3,out_channels=8,kernel_size=3,stride=3)
        self.fc = nn.Sequential(
            nn.Linear(in_features=87616,out_features=2048,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2048,out_features=1024,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=self.hidden_size,bias=True),
            nn.ReLU(),
        )
        self.meta_fc = nn.Sequential(
            nn.Linear(in_features=4,out_features=4,bias=True),
            nn.ReLU(),
        )
    
    def forward(self,x,metadata):
        batch_size, frames, c, h, w = x.shape
        x = x.reshape(batch_size,c,frames,h,w)
        x = self.dcnn(x)
        x = x.view(batch_size,-1)
        x = self.fc(x)
        t = random.randint(0,5)
        metadata = metadata[:,t,:].reshape(batch_size,4)
        metadata = self.meta_fc(metadata.to(torch.float32))
        x = torch.cat((x,metadata),-1).to(torch.float32)
        return x