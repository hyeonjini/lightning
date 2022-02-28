import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl

# python library 'rich' make text looking good
import rich.syntax
import rich.tree

from omegaconf import DictConfig, Dictconfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

def get_logger(name=__name__) -> logging.Logger:
    """_summary_

    Args:
        name (_type_, optional): _description_. Defaults to __name__.

    Returns:
        logging.Logger: _description_
    """
    pass
log = get_logger(__name__)

def extras(config: DictConfig) -> None:
    pass

@rank_zero_only
def print_config(

):
    pass

@rank_zero_only
def log_hyperparameters(

) -> None:
    pass

def finish(

) -> None:
    pass
