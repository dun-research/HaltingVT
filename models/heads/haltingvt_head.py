import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from mmaction.models.heads.base import BaseHead
from mmengine.model.weight_init import trunc_normal_init
from mmaction.evaluation import top_k_accuracy
from mmengine.structures import LabelData


@MODELS.register_module()
class HaltingVTHead(BaseHead):
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
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='AdaHaltCELoss'),
                 init_std: float = 0.02,
                 dropout_ratio: float = 0.0,
                 use_motion_loss = False,
                 use_motion_head = False,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        if use_motion_head:
            self.fc_motion = nn.Linear(self.in_channels, 2)
        
        self.use_motion_head = use_motion_head
        self.use_motion_loss = use_motion_loss
        
        self.loss_name = loss_cls.type

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)
        if self.use_motion_head:
            motion_score = self.fc_motion(x)
            return cls_score, motion_score
        return cls_score
    
    def loss(self, feats, data_samples, **kwargs):
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
        outputs = self(feats)
        
        if self.use_motion_head:
            cls_scores, motion_cls_scores = outputs
            res_kwargs = kwargs.get('res_kwargs', {})
            res_kwargs['motion_cls_scores'] = motion_cls_scores
            return self.loss_by_feat(cls_scores, data_samples, res_kwargs=res_kwargs)
        else:
            cls_scores = outputs   
            return self.loss_by_feat(cls_scores, data_samples, **kwargs)

    def loss_by_feat(self, cls_scores, data_samples, **kwargs):
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        if isinstance(data_samples[0], tuple):
            data_samples_list = []
            for i in range(len(data_samples[0])):
                data_samples_list += [data_samples[0][i], data_samples[1][i]]
            data_samples = data_samples_list
        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()
        
        res_kwargs = kwargs.get('res_kwargs', {})
        if self.use_motion_loss:
            motion_labels = [x.metainfo['motion_label'] for x in data_samples]
            motion_labels = torch.tensor(motion_labels).to(cls_scores.device)
            res_kwargs['motion_labels'] = motion_labels
        
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
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

        if self.loss_name == 'AdaHaltKLCELoss':
            loss_dict = self.loss_cls(cls_scores, labels, res_kwargs=res_kwargs)  
        else:
            loss_cls = self.loss_cls(cls_scores, labels)
            loss_dict = dict(loss_cls=loss_cls)
        # loss_cls may be dictionary or single tensor
        assert isinstance(loss_dict, dict), "loss_dict must be a dict of all sub losses"
        losses.update(loss_dict)
        return losses
    
    def predict(self, feats, data_samples, **kwargs):
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        outputs = self(feats, **kwargs)
        if self.use_motion_head:
            cls_scores, _ = outputs
        else:
            cls_scores = outputs
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores, data_samples):
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_labels`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            prediction = LabelData(item=score)
            pred_label = LabelData(item=pred_label)
            data_sample.pred_scores = prediction
            data_sample.pred_labels = pred_label
        return data_samples