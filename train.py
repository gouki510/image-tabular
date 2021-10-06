from datamodule import MyDataModule
from lit_trainer import Model
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import loggers
from efficientnet_pytorch import EfficientNet
import yaml
import hydra
import os


@hydra.main(config_name="config")
def main(config):
    datamodule = MyDataModule(config)
    model = Model(config)
    #earystopping = EarlyStopping(monitor="train_loss")
    #lr_monitor = callbacks.LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger("logs/")
    loss_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
        mode="min",
    )
    trainer = pl.Trainer(
            max_epochs=config["epoch"],
            logger=tb_logger,
            callbacks=loss_checkpoint,
            gpus=config["gpu"]
        )

    trainer.fit(model,datamodule=datamodule)

if "__main__" == __name__:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()