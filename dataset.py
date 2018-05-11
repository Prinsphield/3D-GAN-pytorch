# -*- coding:utf-8 -*-
# Created Time: 2018/05/10 17:22:38
# Author: Taihong Xiao <xiaotaihong@126.com>

import os
import scipy.ndimage as nd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Config:
    @property
    def data_dir(self):
        # data_dir = '/home/xiaoth/datasets/ModelNet/volumetric_data'
        data_dir = '/gpfs/share/home/1501210096/datasets/ModelNet/volumetric_data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join('train_log')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def model_dir(self):
        model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def img_dir(self):
        img_dir = os.path.join(self.exp_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        return img_dir

    nchw = [16,64,64,64]

    G_lr = 2.5e-3

    D_lr = 1e-5

    step_size = 2000

    gamma = 0.95

    shuffle = True

    num_workers = 5

    max_iter = 20000

config = Config()


class Single(Dataset):
    def __init__(self, filenames, config):
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        voxel = sio.loadmat(self.filenames[idx])['instance']
        voxel = np.pad(voxel, (1,1), 'constant', constant_values=(0,0))
        if self.config.nchw[-1] != 32:
            ratio = self.config.nchw[-1] / 32.
            voxel = nd.zoom(voxel, (ratio, ratio, ratio), mode='constant', order=0)
        return np.expand_dims(voxel.astype(np.float32), 0)

    def gen(self):
        dataloader = DataLoader(self, batch_size=self.config.nchw[0], shuffle=self.config.shuffle, num_workers=self.config.num_workers, drop_last=True)
        while True:
            for data in dataloader:
                yield data


class ShapeNet(object):
    def __init__(self, category, config=config):
        self.category = category
        self.config = config

        self.dict = {True: None, False: None}
        for is_train in [True, False]:
            prefix = os.path.join(self.config.data_dir, category, '30')
            data_dir = prefix + '/train' if is_train else prefix + '/test'
            filenames = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.mat')]
            self.dict[is_train] = Single(filenames, self.config).gen()

    def gen(self, is_train):
        data_gen = self.dict[is_train]
        return data_gen

def test():
    dataset = ShapeNet('chair')
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    for i in range(10):
        if 1 % 2 == 0:
            voxel = next(dataset.gen(True))
        else:
            voxel = next(dataset.gen(False))
        print(i)
        print(voxel.shape)

    pr.disable()
    pr.print_stats()


if __name__ == "__main__":
    test()
