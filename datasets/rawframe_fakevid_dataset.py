import os.path as osp
import random
import copy
import logging
import numpy as np
from random import shuffle
from typing import Callable, List, Optional, Union

import torch
from mmengine.fileio import exists, list_from_file
from mmengine.logging import print_log
from mmengine.dataset.base_dataset import Compose

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmaction.datasets.base import BaseActionDataset


@DATASETS.register_module()
class RawFrameFakeVidDataset(BaseActionDataset):
    """Rawframe dataset for action recognition with fake static video.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        filename_tmpl (str): Template for each filename.
            Defaults to ``img_{:05}.jpg``.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Defaults to False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 1, since raw frames count from 1.
            Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(img=''),
                 filename_tmpl: str = 'img_{:05}.jpg',
                 with_offset: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 1,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 from_same_vid: bool = False,
                 **kwargs) -> None:
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)
        
        self.from_same_vid = from_same_vid
        if not self.from_same_vid:
            self.fake_vid_data_list = self.load_data_list()
            shuffle(self.fake_vid_data_list)
        
        self.pipeline_sampleframes = Compose([pipeline[0]])
        self.pipeline_framedecoder = Compose([pipeline[1]])
        self.pipeline_augs = Compose(pipeline[2:])
        
    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            video_info = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)
            video_info['frame_dir'] = frame_dir
            idx += 1
            if self.with_offset:
                # idx for offset and total_frames
                video_info['offset'] = int(line_split[idx])
                video_info['total_frames'] = int(line_split[idx + 1])
                idx += 2
            else:
                # idx for total_frames
                video_info['total_frames'] = int(line_split[idx])
                idx += 1
            # idx for label[s]
            label = [int(x) for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            if self.multi_class:
                assert self.num_classes is not None
                video_info['label'] = label
            else:
                assert len(label) == 1
                video_info['label'] = label[0]
            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['filename_tmpl'] = self.filename_tmpl
        data_info['motion_label'] = 1   # true video
        
        if self.test_mode:
            return data_info
        
        if self.from_same_vid:
            fake_vid_info = copy.deepcopy(data_info)
        else:
            # get fake video info
            fake_vid_info = copy.deepcopy(self.fake_vid_data_list[idx])

            if idx >= 0:
                fake_vid_info['sample_idx'] = idx
            else:
                fake_vid_info['sample_idx'] = len(self) + idx

        fake_vid_info['modality'] = self.modality
        fake_vid_info['start_index'] = self.start_index
        fake_vid_info['filename_tmpl'] = self.filename_tmpl
        fake_vid_info['motion_label'] = 0   # fake video

        return data_info, fake_vid_info


    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
            
    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if self.test_mode:
            data_info = self.get_data_info(idx)
            return self.pipeline(data_info)
        
        data_info, fake_vid_info = self.get_data_info(idx)
        
        true_vid = self.pipeline(data_info)
        clip_len = true_vid['inputs'].shape[2]
        
        # prepare fake vid
        fake_vid_info = self.pipeline_sampleframes(fake_vid_info)
        fake_vid_frame_idx = random.sample(fake_vid_info['frame_inds'].tolist(), 1)
        fake_vid_info['frame_inds'] = np.array([fake_vid_frame_idx]).reshape(1,)
        fake_vid_info = self.pipeline_framedecoder(fake_vid_info)
        fake_vid_seq = fake_vid_info['imgs'] * clip_len
        fake_vid_info['imgs'] = fake_vid_seq
        fake_vid_info['frame_inds'] = np.array([fake_vid_frame_idx[0] for _ in range(clip_len)])
        fake_vid = self.pipeline_augs(fake_vid_info)
        
        merged_dict = self.merge_data_dict(true_vid, fake_vid)
        return merged_dict

    def merge_data_dict(self, true_vid, fake_vid):
        merged_dict = {}
        inputs_merged = torch.cat([true_vid['inputs'], fake_vid['inputs']], dim=0)
        merged_dict['inputs'] = inputs_merged
        merged_dict['data_samples'] = [true_vid['data_samples'], fake_vid['data_samples']]
        
        return merged_dict