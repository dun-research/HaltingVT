from typing import List, Optional

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmaction.models.losses.base import BaseWeightedLoss
from mmengine.logging import MessageHub


@MODELS.register_module()
class AdaHaltKLCELoss(BaseWeightedLoss):
    """Cross Entropy Loss, Ponder Loss and Motion Loss.

    Args:   
        ponder_token_scale (float): The scale of the ponder token.
        motion_loss_beta (float): The beta of the motion loss.
        loss_weight (float): The weight of the loss.
        class_weight (Optional[List[float]]): The weight of each class.
    """

    def __init__(self,
                 ponder_token_scale = 5e-4,                                                     
                 motion_loss_beta = 0.0,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.loss_weight = loss_weight
        
        self.ponder_token_scale = ponder_token_scale      
        self.motion_loss_beta = motion_loss_beta

    def _forward(self, 
                 cls_score: torch.Tensor, 
                 label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if 'res_kwargs' in kwargs.keys():
            res_kwargs = kwargs.pop('res_kwargs')
        else:
            res_kwargs = {}
        
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)
        
        loss_dict = {'loss_cls': loss_cls}

        # Ponder Loss
        rho_token = res_kwargs.get('rho_token', None)
        if self.ponder_token_scale > 0.:
            assert rho_token is not None, 'rho_token is not provided'
            ponder_loss_token = torch.mean(rho_token) * self.ponder_token_scale
            loss_dict['loss_ponder_token'] = ponder_loss_token
                   
        # Motion loss
        if self.motion_loss_beta > 0.:
            motion_cls_score = res_kwargs.get('motion_cls_scores', None)
            motion_labels = res_kwargs.get('motion_labels', None)
            assert motion_cls_score is not None, 'motion_cls_scores is not provided' 
            assert motion_labels is not None, 'motion_labels is not provided' 
            motion_loss = F.cross_entropy(motion_cls_score, motion_labels)
            loss_dict['loss_motion'] = motion_loss
        
        return loss_dict