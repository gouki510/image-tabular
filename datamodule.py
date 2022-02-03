import cv2
import os
import random
from natsort import natsorted
from glob import glob
import numpy as np
from numpy import False_
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pydicom as dicom
import pytorch_lightning as pl
from albumentations import (
    Compose,
    Equalize,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    Rotate,
    CenterCrop,
)
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import loggers
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    
    def __init__(self,data_dir,img_size,crop_size,data_csv,is_train):
        self.data_dir = data_dir
        self.img_size = img_size
        self.crop_size = crop_size
        self.data_csv = data_csv
        #self.data_csv_r = self.data_csv.reset_index()
        self.is_train = is_train
        self.train_transform, self.test_transform = self._get_transform(self.img_size,self.crop_size)
        
    def _read_image(self,img_path):
        try:
            image = dicom.dcmread(img_path).pixel_array
        except RuntimeError:
            print("img_path:",img_path)
        image = np.stack((image,)*3, axis=-1).astype(np.uint8)
        return image
    
    def _get_img_path(self,img_id):
        if self.is_train:
            img_paths = natsorted(random.sample(glob(os.path.join(self.data_dir,'train',img_id,"*.dcm")),6))
        else:
            img_paths = natsorted(random.sample(glob(os.path.join(self.data_dir,'test',img_id,"*.dcm")),6))
        return img_paths
    
    def _get_label(self,img_id):
        label = self.data_csv.loc[img_id,"FVC"][-1] #最後を予測
        return label

    def _get_metadata(self,img_id):
        metadata = self.data_csv.loc[img_id].sample(6).sort_values("Weeks")
        metadata = metadata.drop(["FVC","Weeks"],axis=1)
        metadata = self._metadata_preprocess(metadata)
        return metadata.values
    
    def _metadata_preprocess(self,metadata):
        if metadata["SmokingStatus"][0]=="Ex-smoker":
            metadata["SS_id"]=0
        elif metadata["SmokingStatus"][0]=="Currently smokes":
            metadata["SS_id"]=1
        else :
            metadata["SS_id"]=-1
        metadata = metadata.drop("SmokingStatus",axis=1)
        if metadata["Sex"][0]=="Male":
            metadata["Sex_id"]=0
        else:
            metadata["Sex_id"]=1
        metadata = metadata.drop("Sex",axis=1)
        return metadata
    
    def _get_transform(self,img_size,crop_size):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = Compose(
            [
                Resize(img_size,img_size),
                #Rotate(limit=5, p=0.2),
                #HorizontalFlip(p=0.5),
                #RandomBrightnessContrast(p=0.2),
                #Equalize(p=0.1),
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
        img_id = self.data_csv.index.values[idx]
        img_paths = self._get_img_path(img_id)
        images = []
        for img_path in img_paths:
            image = self._read_image(img_path)
            if self.is_train:
                image = self.train_transform(image=image)
            else:
                image = self.test_transform(image=image)
            images.append(image["image"])
        images = torch.stack(images)
        metadata = self._get_metadata(img_id)
        label = self._get_label(img_id)
        return images, metadata, label
        
class MyDataModule(pl.LightningDataModule):
    
    def __init__(self,cfg):
        super().__init__()
        self.data_dir = cfg["data_dir"]
        self.img_size = cfg["img_size"]
        self.crop_size = cfg["crop_size"]
        self.data_csv = self.read_csv(cfg["data_csv"])
        self.batch_size = cfg["batch_size"]
    
    def read_csv(self,csv):
        return pd.read_csv(csv,index_col=0)
    
    def prepare_data(self):
        pass      

    def setup(self,stage=None):
        if stage == 'fit' or stage is None:
            train_id, valid_id = train_test_split(self.data_csv.index.unique(), test_size=0.2, shuffle=True)
            self.train_set = MyDataset(self.data_dir,self.img_size,self.crop_size,self.data_csv.loc[train_id],is_train=True)
            self.valid_set = MyDataset(self.data_dir,self.img_size,self.crop_size,self.data_csv.loc[valid_id],is_train=True)
        if stage == "test" or stage is None:
            self.test_set = MyDataset(self.img_size,self.crop_size,self.data_csv,is_train=False)
    
    def train_dataloader(self):
        print("="*100)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=11,
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11,
            pin_memory=True
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11,
            pin_memory=True
        )
    