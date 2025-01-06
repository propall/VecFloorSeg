import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg

"""
l1regularizer
l2regularizer
compute_loss

"""

def l2regularizer(model, alpha=5e-4):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss


def l1regularizer(model, beta=1e-3):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))
    return l1_loss


def compute_loss(pred, true, **kwargs):
    """
    Compute primary loss for the model and prediction score.
    It supports multiple types of loss functions based on the task (e.g., binary classification, regression, or multi-class classification).

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    - Binary Cross-Entropy with Logits(BCEWithLogitsLoss): Used for binary or multi-label classification tasks.
    - Mean Squared Error (MSELoss): Used for regression tasks.


    """
    # Define some loss variables
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    # for func in register.loss_dict.values():
    #     value = func(pred, true)
    #     if value is not None:
    #         return value
    
    # Add L1/L2 regularization to the loss if enabled in the configuration.
    if cfg.model.loss_regularization:
        reg = l2regularizer(kwargs['model'])
    else:
        reg = 0

    # Check if custom loss is registered 
    if cfg.model.loss_fun in register.loss_dict.keys():
        value = register.loss_dict[cfg.model.loss_fun]
        if value.to_be_inited:
            value.init(
                None, pred.device, alpha=cfg.dataset.category,
                reduction=cfg.model.size_average
            )
        return value(pred, true) + reg, pred

    # Calculate loss
    
    ## Classification tasks
    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, ignore_index=255) + reg, pred
        # binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true) + reg, torch.sigmoid(pred)

    ## Regression tasks
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true) + reg, pred

    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

# Computes auxiliary loss for tasks that require an additional objective (e.g., multi-task learning or when there are auxiliary outputs from the model).
# Default auxiliary loss function is BCEWithLogitsLoss
def compute_aux_loss(pred, true, **kwargs):
    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true
    
    # weight = torch.ones_like(pred) * 11.53
    # weight[true == 1] = 0.52 # not partition的标签是1

    # bce_loss = nn.BCEWithLogitsLoss(
    #     weight=weight, reduction=cfg.model.size_average)
    
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    if cfg.model.aux_loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, ignore_index=255), pred # negative log likelihood loss
        
        # binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)

    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

"""
Computes multi-stage loss, such as Dice loss, for segmentation tasks. Multi-stage losses are used when the model produces intermediate outputs that contribute to the final prediction.
"""

def compute_multi_stage_loss(pred, true, **kwargs):
    eps = 1e-5
    pred = pred.unsqueeze(-1) if pred.ndim == 1 else pred
    true = true.unsqueeze(-1) if true.ndim == 1 else true
    if cfg.model.multi_stage_loss_fun == 'dice_loss':
        pred = torch.sigmoid(pred)
        tp = torch.sum(true * pred, dim=1)
        fp = torch.sum(pred, dim=1) - tp
        fn = torch.sum(true, dim=1) - tp
        loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return 1 - loss.sum() / true.shape[0], pred

    else:
        raise ValueError('multi-stage loss func {} not supported'.
                         format(cfg.model.multi_stage_loss_fun))
