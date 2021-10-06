import cv2
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from albumentations import (
    Compose,
    Equalize,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    Rotate,
)
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import loggers
from efficientnet_pytorch import EfficientNet

class Model(pl.LightningModule):
    
    def __init__(self,cfg):
        super().__init__()
        self.backbone = self.load_model(cfg["backbone"])
        self.lr = cfg["lr"]
        self.lr_decay_freq = cfg["lr_decay_freq"] 
        self.embedding = None
        self.dropout = nn.Dropout(0.2)
        self.predict = nn.Linear(in_features=140,out_features=1,bias=True)
        
    def training_step(self,batch,batch_idx):
        images, metadatas, targets = batch
        images = images['image']
        self.embedding = self.dropout(self.backbone(images))
        self.concat = torch.cat((self.embedding,metadatas),1)
        pred = self.predict(self.concat)
        loss = torch.sqrt(torch.mean((pred-targets)**2))
        return {"loss": loss, "preds": pred.clone().detach(), "targets": targets.clone().detach()}
    
    def validation_step(self,batch,batch_idx):
        images, metadatas, targets = batch
        images = images['image']
        self.embedding = self.backbone(images)
        self.concat = torch.cat((self.embedding,metadatas),1)
        pred = self.predict(self.concat)
        loss = torch.sqrt(torch.mean((pred-targets)**2))
        return {"loss": loss, "preds": pred.clone().detach(), "targets": targets.clone().detach()}
    
    def test_step(self,batch,batch_idx):
        images, targets = batch
        images = images['image']
        pred = self.backbone(images)
        loss = torch.sqrt(torch.mean((pred-targets)**2))
        return {"loss": loss, "preds": pred.clone().detach(), "targets": targets.clone().detach()}
    
    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')
    
    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')
    
    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['preds'], out['targets']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(torch.mean((preds-labels)**2))
        self.log(f'{mode}_loss', metrics)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.backbone.parameters(), lr=self.lr, weight_decay=1e-5
        )
        step_lr = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_decay_freq, gamma=0.7
        )
        return [optimizer], [step_lr]

    def load_model(self,backbone):
        if backbone=="EfficientNet":
            model_effnet = EfficientNet.from_name("efficientnet-b2").cuda()
            model_effnet.load_state_dict(torch.load("/home/gominegishi/data/kgl/pet/efficientnet-b2-27687264.pth"))
            model_effnet._fc = nn.Sequential(
                nn.Linear(in_features=1408, out_features=128, bias=True),
                nn.ReLU()
            )

        return model_effnet
            