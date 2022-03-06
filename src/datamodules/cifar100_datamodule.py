from typing import Optional, Tuple

import torch
import pytorch_lightning as pl

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.transforms import transforms

import os

class CIFAR100DataModule(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        data_dir : str = "data/cifar100/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        img_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # init
        self.data_dir = data_dir

        # image size
        

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        #
        self.data_train: Optional[Dataset]
        self.data_val: Optional[Dataset]
        self.data_test: Optional[Dataset]

    @property
    def num_calsses(self) -> int:
        return 100

    def prepare_data(self) -> None:
        """Download data set

        Returns:
            _type_: _description_
        """
        
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.
        Set variables: 'self.data_train', 'self.data_val', 'self.data_test'
        This method is called by 'trainer.fit()' and ' test.fit()'

        Args:
            stage (Optional[str], optional): _description_. Defaults to None.
        """
        train_path = os.path.join(self.data_dir, "train")
        val_path = os.path.join(self.data_dir, "val")
        test_path = os.path.join(self.data_dir, "val")

        self.data_train = ImageFolder(train_path, transform=self.transforms)
        self.data_val = ImageFolder(val_path, transform=self.transforms)
        self.data_test = ImageFolder(test_path, transform=self.transforms)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )