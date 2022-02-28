from typing import Any, List

import torch
import pytorch_lightning as pl

from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from components.simple_dense_net import SimpleDenseNet

class CIFAR100Module(pl.LightningModule):
    """
    Example of LightningModule for CIFAR-100 classification.

    5 section
        - Computation (init).
        - Train loop (training_step).
        - Validation loop (validation_step).
        - Test loop (test_step).
        - Optimizers (configure_optimizers).

    """

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # architecture
        self.model = SimpleDenseNet(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.val_acc_best = MaxMetric()

    
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        return loss, preds, y
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        """_summary_

        Args:
            outputs (List[Any]): a list of dicts returned from 'training_step()'
        """
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss ,preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, porg_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute() # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, porg_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
    
    def configure_optimizers(self):

        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )




