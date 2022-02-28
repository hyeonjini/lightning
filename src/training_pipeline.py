import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer, 
    seed_everything
)

from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

# init log

def train(config: DictConfig) -> Optional[float]:
    """_summary_

    Args:
        config (DictConfig): _description_

    Returns:
        Optional[float]: _description_
    """

    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    
    # Convert relative ckpt path to absolute path.
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.jin(
            hydra.utils.get_original_cwd(), ckpt_path
        )
    
    # Init lightning datamodule
    
    print(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    print(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks

    # Init lightning loggers

    # Init lightning trainer
    print(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        #callbacks=callbacks,
        #logger=logger,
        _convert_="partial"
    )

    # Train the model
    if config.get("train"):
        print("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not fund! "
            "Make sure the `optimized_metric` in `hparamas_search` config is correct! "
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model

    # "Make sure everything closed properly"
    print("Finalizing!")
    

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.trainer.get("train"):
        print(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    return score

    