import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.utils.data as data
from models.crapcn import CRAPCN_mvp, CRAPCN_mvp_d
import numpy as np
import logging
import os
import sys
import h5py
from tqdm import tqdm

from utils.mvp_utils import *



class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = './data/MVP/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

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
        partial = torch.from_numpy((self.input_data[index]))

        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            return label, partial, complete
        else:
            return partial


def test():
    dataset_test = MVP_CP(prefix='val')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=8)
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model = CRAPCN_mvp() # or 'CRAPCN_mvp_d'
    net = torch.nn.DataParallel(model)
    net.cuda()
    checkpoint = torch.load('path/ckpt-best.pth')
    net.load_state_dict(checkpoint['model'])
    net.eval()

    metrics = ['cd_p', 'cd_t', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([16, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 150 * 26
    novel_cat_num = torch.ones([8, 1], dtype=torch.float32).cuda() * 50 * 26
    cat_num = torch.cat((cat_num, novel_cat_num), dim=0)
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft', 
                'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
    logging.info('Testing...')
   
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader_test)):
            label, inputs_cpu, gt_cpu = data

            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            #inputs = inputs.transpose(2, 1).contiguous()
            # result_dict = net(inputs, gt, is_training=False, mean_feature=mean_feature)
            output = net(inputs)[-1]
            cd_p, cd_t, f1 = calc_cd(output, gt, calc_f1=True)
            result_dict = dict()
            result_dict['cd_p'] = cd_p
            result_dict['cd_t'] = cd_t
            result_dict['f1'] = f1

            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] += result_dict[m][int(j)]

    

        logging.info('Loss per category:')
        category_log = ''
        for i in range(16):
            category_log += '\ncategory name: %s' % (cat_name[i])
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m == 'f1' else 10000
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
        logging.info(category_log)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    log_dir = 'mvp'
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()