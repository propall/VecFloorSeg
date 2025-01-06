import glob
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Union

import torch

from torch_geometric.graphgym.config import cfg

MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
SCHEDULER_STATE = 'scheduler_state'

"""
Fns in this script
load_ckpt       : Loads the model checkpoint at a given epoch.
save_ckpt       : Saves the model checkpoint at a given epoch.
remove_ckpt     : Removes the model checkpoint at a given epoch.
clean_ckpt      : Removes all but the last model checkpoint.
get_ckpt_dir    : Returns the directory path where checkpoints are stored.
get_ckpt_path   : Returns the path to the checkpoint file for a given epoch.
get_ckpt_epochs : Returns a sorted list of available checkpoint epochs.
get_ckpt_epoch  : Returns the actual checkpoint epoch number based on the input epoch.

"""


def load_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = -1,
    prefix = None
) -> int:
    """
    Loads the model checkpoint at a given epoch.
    
    Args:
        model: The PyTorch model to load the checkpoint for.
        optimizer: The optimizer to load the state for (optional).
        scheduler: The learning rate scheduler to load the state for (optional).
        epoch: The epoch to load the checkpoint from (default: -1, latest epoch).
        prefix: The prefix for the checkpoint file (optional).
        
    Returns:
        The next epoch number after loading the checkpoint.
    """
    epoch = get_ckpt_epoch(epoch)
    path = get_ckpt_path(epoch, prefix)

    if not osp.exists(path):
        return 0

    ckpt = torch.load(path, map_location=torch.device(cfg.device))
    model.load_state_dict(ckpt[MODEL_STATE])
    if optimizer is not None and OPTIMIZER_STATE in ckpt:
        optimizer.load_state_dict(ckpt[OPTIMIZER_STATE])
    if scheduler is not None and SCHEDULER_STATE in ckpt:
        scheduler.load_state_dict(ckpt[SCHEDULER_STATE])

    return epoch + 1


def save_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    prefix = ''
):
    """
    Saves the model checkpoint at a given epoch.
    
    Args:
        model: The PyTorch model to save the checkpoint for.
        optimizer: The optimizer to save the state for (optional).
        scheduler: The learning rate scheduler to save the state for (optional).
        epoch: The current epoch number.
        prefix: The prefix for the checkpoint file (optional).
    """
    ckpt: Dict[str, Any] = {}
    ckpt[MODEL_STATE] = model.state_dict()
    if optimizer is not None:
        ckpt[OPTIMIZER_STATE] = optimizer.state_dict()
    if scheduler is not None:
        ckpt[SCHEDULER_STATE] = scheduler.state_dict()

    os.makedirs(get_ckpt_dir(), exist_ok=True)
    torch.save(ckpt, get_ckpt_path(get_ckpt_epoch(epoch), prefix=prefix))


def remove_ckpt(epoch: int = -1):
    """
    Removes the model checkpoint at a given epoch.
    
    Args:
        epoch: The epoch to remove the checkpoint for (default: -1, latest epoch).
    """
    os.remove(get_ckpt_path(get_ckpt_epoch(epoch)))


def clean_ckpt():
    """
    Removes all but the last model checkpoint.
    """
    for epoch in get_ckpt_epochs()[:-1]:
        os.remove(get_ckpt_path(epoch))


###############################################################################


def get_ckpt_dir() -> str:
    """
    Returns the directory path where checkpoints are stored.
    
    Returns:
        The checkpoint directory path.
    """
    return osp.join(cfg.run_dir, 'ckpt')


def get_ckpt_path(epoch: Union[int, str], prefix='') -> str:
    """
    Returns the path to the checkpoint file for a given epoch.
    
    Args:
        epoch: The epoch number or a string representing the epoch.
        prefix: The prefix for the checkpoint file (optional).
        
    Returns:
        The path to the checkpoint file.
    """
    return osp.join(get_ckpt_dir(), f'{prefix}{epoch}.ckpt')


def get_ckpt_epochs() -> List[int]:
    """
    Returns a sorted list of available checkpoint epochs.
    
    Returns:
        A sorted list of checkpoint epochs.
    """
    paths = glob.glob(get_ckpt_path('*'))
    return sorted([int(osp.basename(path).split('.')[0]) for path in paths])


def get_ckpt_epoch(epoch: int) -> int:
    """
    Returns the actual checkpoint epoch number based on the input epoch.
    
    Args:
        epoch: The input epoch number. If negative, it counts from the latest epoch.
        
    Returns:
        The actual checkpoint epoch number.
    """
    if epoch < 0:
        epochs = get_ckpt_epochs()
        epoch = epochs[epoch] if len(epochs) > 0 else 0
    return epoch
