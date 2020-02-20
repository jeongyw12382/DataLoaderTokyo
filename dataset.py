from __future__ import print_function, division
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.io import loadmat
import logging
import copy
from tqdm import tqdm
from PIL import Image
import numpy as np
import fnmatch
import csv

data_path_from_home = '/Data/tokyoTimeMachine'
data_path = os.environ['HOME'] + data_path_from_home

image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

class TokyoDataSet(Dataset):

    def __init__(self, type='train', mode='db', root_dir = data_path, transforms=image_transform):

        self.logger = self.generate_logger(type)
        self.type = type
        self.logger.info('Loading Tokyo ' + ('Train' if type == 'train' else 'Val') + ' Matrix')
        mat_file = '/tokyoTM_train.mat' if type == 'train' else '/tokyoTM_val.mat'
        mat = loadmat(os.path.join(root_dir + mat_file))['dbStruct'][0][0]
        self.root_dir = root_dir
        self.image_dir = self.root_dir + '/images'
        adder = 0 if mode == 'db' else 3
        self.length = len(mat[adder + 1])
        self.data = [{} for _ in range(self.length)]
        self.transforms = transforms
        self.utm = [[mat[adder+2][0][i], mat[adder+2][1][i]] for i in range(self.length)]

        for idx in tqdm(range(self.length)):
            self.data[idx]['filename'] = mat[adder + 1][idx][0][0]
            self.data[idx]['utm_coordinate'] = (mat[adder + 2][0][idx], mat[adder + 2][1][idx])
            self.data[idx]['timestamp'] = mat[adder + 3][0][idx]
            self.data[idx]['image'] = os.path.join(self.image_dir, self.data[idx]['filename'])

            if mode == 'query':
                self.data[idx]['pos'] = [-1] * 10
                self.data[idx]['neg'] = [-1] * 10

        self.logger.info('Done')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        ret = copy.deepcopy(self.data[idx])
        ret['image'] = self.transforms(Image.open(self.data[idx]['image']))
        return ret

    def generate_logger(self, type):
        logger_name = 'trainData' if type == 'train' else 'valData'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def set(self, idx, key, val):
        self.data[idx][key] = val


class TokyoTrainDataSet(TokyoDataSet):

    def __init__(self, mode='db', GT_from_file=False):
        super().__init__(type='train', mode=mode)


class TokyoValDataSet(TokyoDataSet):

    def __init__(self, mode='db', GT_from_file=False):
        super().__init__(type='val', mode=mode)


class Tokyo247(Dataset):

    def __init__(self, root_dir=data_path, transform=image_transform):
        self.logger = self.generate_logger
        self.root_dir = root_dir
        self.image_dir = root_dir + '/247query_v3'
        self.matname = np.sort(fnmatch.filter(os.listdir(self.image_dir), '*.csv'))
        self.imname = np.sort(fnmatch.filter(os.listdir(self.image_dir), '*.jpg'))
        self.length = len(self.imname)
        self.data = np.array([{} for _ in range(self.length)])
        self.transforms = image_transform

        for idx in tqdm(range(self.length)):
            f = open(self.image_dir + '/' + self.matname[idx])
            mat = csv.reader(f,  delimiter=',')
            mat = list(mat)[0]
            f.close()

            self.data[idx]['filename'] = mat[0]
            self.data[idx]['utm_coordinate'] = (mat[7], mat[8])
            self.data[idx]['image'] = os.path.join(self.image_dir, self.data[idx]['filename'])


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        ret = copy.deepcopy(self.data[idx])
        ret['image'] = self.transforms(Image.open(self.data[idx]['image']))
        return ret


    def generate_logger(self):
        logger_name = 'testData'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

