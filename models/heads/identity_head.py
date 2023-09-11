from mmengine.model.weight_init import trunc_normal_init
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from mmaction.models.heads.base import BaseHead
from mmaction.evaluation import top_k_accuracy



@MODELS.register_module()
class IdentityHead(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        dropout_ratio (float): Probability of dropout layer.
            Defaults to : 0.0.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 **kwargs) -> None:
        super().__init__(num_classes=400, in_channels=768, loss_cls=loss_cls, **kwargs)

        self.head = nn.Identity()

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        return

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        return self.head(x)


    def loss(self, feats,
             data_samples,
             **kwargs):
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples, **kwargs)

    def loss_by_feat(self, cls_scores,
                     data_samples,
                     **kwargs):
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses