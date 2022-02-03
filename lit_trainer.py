import cv2
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
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
from models import CNNLSTM,CNN,DCNN

class Model(pl.LightningModule):
    
    def __init__(self,cfg):
        super().__init__()
        self.lr = cfg["lr"]
        self.lr_decay_freq = cfg["lr_decay_freq"] 
        self.num_layer = cfg["num_layer"]
        self.hidden_size = cfg["hidden_size"]
        self.loss_type = cfg["loss_type"]
        self.model_type = cfg["model_type"]
        if  self.model_type == "CNN":
            self.embedder = CNN(self.hidden_size)
        elif self.model_type == "CNNLSTM":
            self.embedder = CNNLSTM(self.num_layer,self.hidden_size)
        elif self.model_type == "3DCNN":
            self.embedder = DCNN(self.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.predict = nn.Sequential(
            nn.Linear(in_features=self.hidden_size+4,out_features=128,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=1,bias=True),
        )
        
    def training_step(self,batch,batch_idx):
        images, metadatas, targets = batch
        images = images
        self.embedding = self.embedder(images,metadatas)
        pred = self.predict(self.embedding)
        loss = self.loss_fn(pred,targets,self.loss_type)
        return {"loss": loss, "preds": pred.clone().detach(), "targets": targets.clone().detach()}
    
    def validation_step(self,batch,batch_idx):
        images, metadatas, targets = batch
        self.embedding = self.embedder(images,metadatas)
        pred = self.predict(self.embedding)
        loss = self.loss_fn(pred,targets,self.loss_type)
        return {"loss": loss, "preds": pred.clone().detach(), "targets": targets.clone().detach()}
    
    def test_step(self,batch,batch_idx):
        images, metadatas = batch
        self.embedding = self.embedder(images,metadatas)
        pred = self.predict(self.embedding)
        return { "preds": pred.clone().detach()}
    
    def loss_fn(self,pred,targets,loss_type):
        if loss_type=="L2":
            return torch.sqrt(torch.mean((pred-targets)**2))
        elif loss_type=="L1":
            return torch.mean(torch.abs(pred-targets))
    
    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')
    
    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')
    
    def __share_epoch_end(self, outputs, mode):
        preds = []
        targets = []
        for out in outputs:
            pred, label = out['preds'], out['targets']
            preds.append(pred)
            targets.append(label)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        metrics = self.loss_fn(preds,targets,self.loss_type)
        self.log(f'{mode}_loss', metrics)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.embedder.parameters(), lr=self.lr, weight_decay=1e-5
        )
        step_lr = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_decay_freq, gamma=0.7
        )
        return [optimizer], [step_lr]
