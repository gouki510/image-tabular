import cv2
import os
from numpy import False_
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
from sklearn.model_selection import train_test_split

BASE_DIR = "/home/gominegishi/data/kgl/pet/petfinder-pawpularity-score"

class MyDataset(Dataset):
    
    def __init__(self,img_size,data_csv,is_train):
        self.img_size = img_size
        self.data_csv = data_csv
        self.data_csv_r = self.data_csv.reset_index()
        self.is_train = is_train
        self.train_transform, self.test_transform = self._get_transform(self.img_size)
        
    def _read_image(self,img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
    
    def _get_img_path(self,img_id):
        if self.is_train:
            img_path = os.path.join(BASE_DIR,'train',f"{img_id}.jpg")
        else:
            img_path = os.path.join(BASE_DIR,'train',f"{img_id}.jpg")
        return img_path
    
    def _get_label(self,img_id):
        label = self.data_csv.loc[img_id,"Pawpularity"]
        return label

    def _get_metadata(self,img_id):
        metadata = self.data_csv.loc[img_id]
        metadata = metadata.drop("Pawpularity")
        return metadata.values

    
    def _get_transform(self,img_size):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = Compose(
            [
                Resize(img_size,img_size),
                Rotate(limit=5, p=0.2),
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.2),
                Equalize(p=0.1),
                Normalize(mean=mean,std=std),
                ToTensorV2()
            ]
        )
        test_transform = Compose(
            [
                Resize(img_size,img_size),
                Normalize(mean=mean,std=std),
                ToTensorV2()
            ]
        )
        return train_transform, test_transform
        
    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self,idx):
        img_id = self.data_csv_r["Id"].values[idx]
        img_path = self._get_img_path(img_id)
        image = self._read_image(img_path)
        if self.is_train:
            image = self.train_transform(image=image)
        else:
            image = self.test_transform(image=image)
        metadata = self._get_metadata(img_id)
        label = self._get_label(img_id)
        return image, metadata, label
        
class MyDataModule(pl.LightningDataModule):
    
    def __init__(self,cfg):
        super().__init__()
        self.img_size = cfg["img_size"]
        self.data_csv = self.read_csv(cfg["data_csv"])
        self.batch_size = cfg["batch_size"]
    
    def read_csv(self,csv):
        return pd.read_csv(csv,index_col=0)
    
    def prepare_data(self):
        pass      

    def setup(self,stage=None):
        if stage == 'fit' or stage is None:
            train_x, valid_x = train_test_split(self.data_csv, test_size=0.2, shuffle=True)
            self.train_set = MyDataset(self.img_size,train_x,is_train=True)
            self.valid_set = MyDataset(self.img_size,valid_x,is_train=False)
        if stage == "test" or stage is None:
            self.test_set = MyDataset(self.img_size,self.data_csv,is_train=False)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    