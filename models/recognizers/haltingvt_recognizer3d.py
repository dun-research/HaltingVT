import os
import time
import torch
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType, OptSampleList
from mmaction.models.recognizers.base import BaseRecognizer


@MODELS.register_module()
class HaltingVTRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""
    def __init__(self,
                backbone: ConfigType,
                cls_head: OptConfigType = None,
                neck: OptConfigType = None,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                vis_mode = False,
                ) -> None:
        super().__init__(backbone, cls_head, neck, train_cfg, test_cfg, data_preprocessor)
        
        self.vis_mode = vis_mode

    def extract_feat(self,
                     inputs: Tensor,
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])

        # Check settings of test
        if test_mode:
            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat, _, _ = self.backbone(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views
                # recursively traverse feats until it's a tensor, then concat

                def recursively_cat(feats):
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x = recursively_cat(feats)
                else:
                    x = torch.cat(feats)
            else:
                x, res_kwargs = self.backbone(inputs)
                if self.with_neck:
                    x, _ = self.neck(x)

            return x, res_kwargs, loss_predict_kwargs
        else:
            # Return features extracted through backbone
            x, res_kwargs = self.backbone(inputs)
            return x, res_kwargs

    def loss(self, inputs: torch.Tensor, 
             data_samples, 
             test_mode=False,
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            dict: A dictionary of loss components.
        """
        feats, res_kwargs = self.extract_feat(inputs,
                                        data_samples=data_samples,
                                        test_mode=test_mode)
        loss_cls = self.cls_head.loss(feats, data_samples=data_samples, res_kwargs=res_kwargs)
        return loss_cls

    def predict(self, 
                inputs, 
                data_samples,
                **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            List[``ActionDataSample``]: Return the recognition results.
            The returns value is ``ActionDataSample``, which usually contains
            ``pred_scores``. And the ``pred_scores`` usually contains
            following keys.

                - item (torch.Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        feats, res_kwargs, _ = self.extract_feat(inputs, 
                                                data_samples, 
                                                test_mode=True)
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions
    
    def _forward(self,
                 inputs: torch.Tensor,
                 **kwargs):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        Returns:
            Union[tuple, torch.Tensor]: Features from ``backbone`` or ``neck``
            or ``head`` forward.
        """
        feats, _, _ = self.extract_feat(inputs, test_mode=True)
        return feats
