<div align='center'>
<h1>(AAAI'24) CRA-PCN: Point Cloud Completion with Intra- and Inter-level Cross-Resolution Transformers </h1>
</div>

![example](./vis.png) 

## [CRA-PCN]

This repo contains a PyTorch implementation for **CRA-PCN: Point Cloud Completion with Intra- and Inter-level Cross-Resolution Transformers** (AAAI'24). 
[[**arXiv**]](https://arxiv.org/abs/2401.01552) [[**AAAI**]](https://ojs.aaai.org/index.php/AAAI/article/view/28268)

## [News]
**[2024-03-09]**  We add a new seed generator implemented with [Deconvolution](https://github.com/AllenXiangX/SnowflakeNet). 

**[2024-03-09]**  We add training and testing codes for MVP dataset. 

## [Installation]
❗Tips: If you have a configured virtual environment for [SeedFormer](https://github.com/hrzhou2/seedformer) 
(or [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), [PoinTr](https://github.com/yuxumin/PoinTr)), you can reuse it instead of installing a new one.
### Requirements
Our models have been tested on the configuration below:
- python == 3.6.13
- PyTorch == 1.10.1
- CUDA == 12.2
- numpy == 1.19.5
- open3d ==  0.9.0.0

Step 1. Install requirements:
```
pip install -r requirements.txt
```

Step 2. Compile the C++ extension modules:
```
sh install.sh
```


## [Data preparation]
### PCN dataset
❗Tips: If you already have PCN dataset, you should change the data path in train_pcn.py and test_pcn.py:
```
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH   =  './data/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH  =  './data/PCN/%s/complete/%s/%s.pcd'
```
Otherwise, you need to download PCN dataset from [here](https://gateway.infinitescript.com/s/ShapeNetCompletion), 
and then unzip it and put it under ./data.

### ShapeNet-55/34 dataset
❗Tips: If you already have ShapeNet-55/34 dataset, you should change the data path in train_shapenet55.py:
```
__C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH    =  './data/ShapeNet55-34/ShapeNet-55/'
__C.DATASETS.SHAPENET55.N_POINTS              =  2048      # don't change this line
__C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH  =  './data/ShapeNet55-34/shapenet_pc/%s'
```
and change the data path in train_shapenet34.py:
```
__C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH    =  './data/ShapeNet55-34/ShapeNet-34/'
__C.DATASETS.SHAPENET55.N_POINTS              =  2048      # don't change this line
__C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH  =  './data/ShapeNet55-34/shapenet_pc/%s'
```

Otherwise, you need to download ShapeNet-55/34 dataset from [here](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md), and then unzip it and put it under ./data.

### MVP dataset
You can download MVP dataset from this [link](https://mvp-dataset.github.io/), and put these two .h5 files in MVP folder.
The input & output resolution is 2048.

❗After data preparation, the overall directory structure should be:
```
│CRA-PCN/
├──datasets/
├──data/
│   ├──ShapeNet55-34/
│   ├──PCN/
│   ├──MVP/
│   │   ├──MVP_Test_CP.h5
│   │   ├──MVP_Train_CP.h5
├──.......
```



## [Training & Testing]

### Training & Testing on PCN dataset
Training:
```
python train_pcn.py
```

The training log will be saved at:
```
__C.DIR.OUT_PATH  =  'results/'  # line 88
```
Here, we provide a pretrained weight:

| Dataset | Weight  | Log |
|  ----  | ----  |  ----  |
|  PCN  | [url](https://github.com/EasyRy/CRA-PCN/blob/main/pretrain/pcn/ckpt-best.pth)  | [url](https://github.com/EasyRy/CRA-PCN/blob/main/pretrain/pcn/log.txt) |

❗Note: We have refactored our codes after the acceptance of AAAI'24 and retrained the model on 6x Nvidia GTX 1080 Ti graphic cards with a batch size of 60.

Testing:
```
python test_pcn.py
```
The testing results will be saved at:
```
__C.DIR.TEST_PATH  =  'test/cra-pcn'  # line 80
```

### Training on ShapeNet-55 dataset
Training 
```
python train_shapenet55.py 
```
The training log will be saved at:
```
__C.DIR.OUT_PATH  =  'results/shapenet55' # line 76
```
### Training on ShapeNet-34 dataset
Training:
```
python train_shapenet34.py
```
The training log will be saved at:
```
__C.DIR.OUT_PATH  =  'results/shapenet34' # line 76
```

### Testing on ShapeNet-55/34/Unseen-21 dataset
Testing example:
```
### Testing on ShapeNet-55

### mode = [easy, median, hard]

### _C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH = './data/ShapeNet55-34/ShapeNet-55/'

python test_shapenet.py --pretrained ./pretrain/shapenet/shapenet55.pth --mode easy


### Testing on ShapeNet-34

### mode = [easy, median, hard]

### _C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH = './data/ShapeNet55-34/ShapeNet-34/'

python test_shapenet.py --pretrained ./pretrain/shapenet/shapenet34.pth --mode easy


### Testing on ShapeNet-Unseen21

### mode = [easy, median, hard]

### _C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH = './data/ShapeNet55-34/ShapeNet-Unseen21/'

python test_shapenet.py --pretrained ./pretrain/shapenet/shapenet34.pth --mode easy

```
Please refer to [PoinTr](https://github.com/yuxumin/PoinTr) for more details.

### Training & Testing on MVP dataset
Training 
```
python train_mvp.py 
```
The training log will be saved at:
```
__C.DIR.OUT_PATH  = 'results/mvp_result' # line 143
```

Testing 
```
python test_mvp.py 
```
## [Some details about training & testing]

### Can't reproduce the results
Please refer to [here](https://github.com/EasyRy/CRA-PCN/tree/main/models).

### Training/testing configuration  

You can modify configuration for training/testing in main_xxx.py (e.g., PCNConfig).
Note that, the number of GPUs can be changed at the beginning of train_xxx.py, like:
```
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
```
This idea is borrowed from [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet).


### About manager.py 

This file is used to control the training/testing process, where *Manager* is applied for PCN dataset, *Manager_shapenet55* is applied for ShapeNet-55/34, and *Manager_mvp* is applied for MVP dataset. This idea is borrowed from [SeedFormer](https://github.com/hrzhou2/seedformer).


### Why testing results are unstable?  

It is a common phenomenon due to the randomness of farthest point sampling.


## [Acknowledgement]
This repo is heavily based on [SeedFormer](https://github.com/hrzhou2/seedformer), [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), [GRNet](https://github.com/hzxie/GRNet), [VRCNet](https://github.com/paul007pl/VRCNet), and [PoinTr](https://github.com/yuxumin/PoinTr).
We thank for their excellent works.

## [Citation]
```
@inproceedings{rong2024cra,
  title={CRA-PCN: Point Cloud Completion with Intra-and Inter-level Cross-Resolution Transformers},
  author={Rong, Yi and Zhou, Haoran and Yuan, Lixin and Mei, Cheng and Wang, Jiahao and Lu, Tong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4676--4685},
  year={2024}
}
```
