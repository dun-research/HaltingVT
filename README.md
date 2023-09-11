# HaltingVT：Adaptive Token Halting Transformer for Efficient Video Recognition

This is an official pytorch implementation of paper **_"HaltingVT：Adaptive Token Halting Transformer for Efficient Video Recognition"_**. This repository provides code for video recognition. Also, the evaluation code and pretrained weights are available to facilitate the reproduction of the paper's results. This repository is based on [MMAction2](https://github.com/open-mmlab/mmaction2). 

- For HaltingVT code implementation, please refer to `models/backbones/haltingvt_model.py`, including the implementation of Glimpser and Motion Loss as well; 
- For details about the data preprocessing operation of Motion Loss, see `datasets/rawframe_fakevid_dataset.py`.
- Settings of parameters are all in `configs/`.


## Installation
Requirements
 + python 3.6+
 + torch>=1.7.1
 + mmcv==2.0.0 and mmaction2==1.0.0

You can install all the dependencies by 
```
    pip install -r requirements.txt
```
install [mmcv](https://github.com/open-mmlab/mmcv) and [mmaction2](https://github.com/open-mmlab/mmaction2) following the official instruction.


## Dataset Preparation

#### 1. Download videos
  + Please download datasets following the official guidlines.
  + Currently supported datasets: [Mini-Kinetics](https://www.deepmind.com/open-source/kinetics) and [ActivityNet-v1.3](http://activity-net.org/download.html).


#### 2. Prepare annotation files
  + **Mini-Kinetics：** Get splits and annotation files from [https://github.com/mengyuest/AR-Net](https://github.com/mengyuest/AR-Net);
  + **ActivityNet-v1.3：** Use the [official splits and annotation files](http://activity-net.org/download.html)；Preprocess datas with scripts in MMAction2 following the [instruction](tools/data/activitynet/README.md).
  + Set the data root and annotation file path in `configs/minik/minik_roots.py` and `configs/activitynet/anet_roots.py`, respectively.


## Evaluation
We provide code and pretrained weights to reproduce the experiments in the paper.

### 1. Pretrained Weights
We have provided our models on google drive [Google Drive](https://drive.google.com/drive/folders/1ON9PGhGmvTz-y2Qb9mgbSTp54TChCajY?usp=sharing).

### 2. Run Evaluation

```
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=8       --master_port=29005 eval.py \
      path_to_config \
      path_to_pretrained_weights \
      --launcher pytorch
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact us for further details
The code for training is not included in this repository. We can not release the training code publicly for IP reasons. If you need the training code or have any questions, please contact us.
