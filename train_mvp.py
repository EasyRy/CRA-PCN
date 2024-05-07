import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import argparse
import numpy as np
import torch
import json
import time
import utils.data_loaders
from easydict import EasyDict as edict
from importlib import import_module
from pprint import pprint
from manager import Manager_MVP
import torch.utils.data as data
import h5py
import math
import transforms3d
from models.crapcn import CRAPCN_mvp, CRAPCN_mvp_d

TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]


# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing MVP', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=True, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
args = parser.parse_args()


# MVP dataset
# input: 2048
# output: 2048
def RandomMirrorPoints(ptcloud,rnd_value):
    trfm_mat = transforms3d.zooms.zfdir2mat(1)
    trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]),
                            trfm_mat)
    trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]),
                            trfm_mat)
    if rnd_value <= 0.25:
        trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        trfm_mat = np.dot(trfm_mat_z, trfm_mat)
    elif rnd_value > 0.25 and rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
        trfm_mat = np.dot(trfm_mat_x, trfm_mat)
    elif rnd_value > 0.5 and rnd_value <= 0.75:
        trfm_mat = np.dot(trfm_mat_z, trfm_mat)

    ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
    return ptcloud


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train", aug=False):
        if prefix=="train":
            self.file_path = './data/MVP/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.aug = aug
        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.prefix == "train" and self.aug:
            a = np.random.uniform(0, 1)
            partial = RandomMirrorPoints(self.input_data[index], a)
            gt_cloud = RandomMirrorPoints(self.gt_data[index // 26], a)
        else:
            partial = self.input_data[index]
            gt_cloud = self.gt_data[index // 26]

        partial = torch.from_numpy(partial)
        gt_cloud = torch.from_numpy(gt_cloud)
        
        label = (self.labels[index])
        return label, partial, gt_cloud
        
# Configuration for MVP
def MVPConfig():

    __C                                              = edict()
    cfg                                              = __C

    #
    # Dataset Config
    #
    __C.DATASETS                                     = edict()
    __C.DATASETS.COMPLETION3D                        = edict()
    __C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
    __C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
    __C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
    __C.DATASETS.SHAPENET                            = edict()
    __C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
    __C.DATASETS.SHAPENET.N_RENDERINGS               = 8
    __C.DATASETS.SHAPENET.N_POINTS                   = 2048
    

    __C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        =  '../data/PCN/%s/partial/%s/%s/%02d.pcd'
    __C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       =  '../data/PCN/%s/complete/%s/%s.pcd'

    #
    # Dataset
    #
    __C.DATASET                                      = edict()
    # Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
    __C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
    __C.DATASET.TEST_DATASET                         = 'ShapeNet'

    #
    # Constants
    #
    __C.CONST                                        = edict()

    __C.CONST.NUM_WORKERS                            = 8
    __C.CONST.N_INPUT_POINTS                         = 2048

    #
    # Directories
    #

    __C.DIR                                          = edict()
    __C.DIR.OUT_PATH                                 = 'results/mvp_result'
    __C.DIR.TEST_PATH                                = 'test/MVP'
    __C.CONST.DEVICE                                 = '0, 1'
    # __C.CONST.WEIGHTS                                = None # 'ckpt-best.pth'  # specify a path to run test and inference

    #
    # Network
    #
    __C.NETWORK                                      = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                     = [2, 2, 1, 8] # 16384
    __C.NETWORK.KP_EXTENTS                           = [0.1, 0.1, 0.05, 0.025] # 16384
    #
    # Train
    #
    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 44
    __C.TRAIN.N_EPOCHS                               = 200
    __C.TRAIN.SAVE_FREQ                              = 25
    __C.TRAIN.LEARNING_RATE                          = 0.0001
    __C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
    __C.TRAIN.LR_DECAY_STEP                          = 50
    __C.TRAIN.WARMUP_STEPS                           = 200
    __C.TRAIN.WARMUP_EPOCHS                          = 20
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0.001
    __C.TRAIN.LR_DECAY                               = 150

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg


def train_net(cfg):

    # return 
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    train_dataset = MVP_CP(prefix='train')
    val_dataset = MVP_CP(prefix='val')

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    #collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  #collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)


    
    # Set up folders for logs and checkpoints
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # save config file
    pprint(cfg)
    config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
    with open(config_filename, 'w') as file:
        json.dump(cfg, file, indent=4, sort_keys=True)

    # Save Arguments
    torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

    #######################
    # Prepare Network Model
    #######################)
    model = CRAPCN_mvp() # or 'CRAPCN_mvp_d'

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    
    manager = Manager_MVP(model, cfg)

    manager.train(model, train_data_loader, val_data_loader, cfg)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    
    seed = 1128
    set_seed(seed)
    
    print('cuda available ', torch.cuda.is_available())

    cfg = MVPConfig()

    train_net(cfg)
    