import os
import sys
import json
import random

import PIL
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

class VG(data.Dataset):

    def __init__(self, type, args, mode,
                 image_dir, anno_path, labels_path,
                 transform=None):

        assert mode in ('train', 'val')

        self.mode = mode
        self.name = 'VG'
        self.transform = transform

        self.img_dir = image_dir
        self.imgName_path = anno_path
        self.img_names = open(self.imgName_path, 'r').readlines()
        # 限制训练样本数量
        if mode == 'train':
            self.img_names = self.img_names[:1000]
        if mode == 'val':
            self.img_names = self.img_names[:4000]
        self.labels_path = labels_path
        _ = json.load(open(self.labels_path, 'r'))
        self.labels = np.zeros((len(self.img_names), 200)).astype(np.int32)
        for i in range(len(self.img_names)):
            self.labels[i][_[self.img_names[i][:-1]]] = 1
        self.Y = self.labels

    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        if self.transform:
           input = self.transform(input)
        return input, self.labels[index]

    def __len__(self):
        return len(self.img_names)

def getPairIndexes(labels):

    res = []
    for index in range(labels.shape[0]):
        tmp = []
        for i in range(labels.shape[1]):
            if labels[index, i] > 0:
                tmp += np.where(labels[:, i] > 0)[0].tolist()

        tmp = set(tmp)
        tmp.discard(index)
        res.append(np.array(list(tmp)))

    return res

def get_VG(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    randomCropList = [transforms.RandomCrop(Size) for Size in [448, 384, 320]]
    train_data_transform = transforms.Compose(
        [transforms.Resize((448, 448), interpolation=PIL.Image.BICUBIC),
         transforms.RandomChoice(randomCropList),
         transforms.Resize((448, 448), interpolation=PIL.Image.BICUBIC),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize])

    test_data_transform = transforms.Compose(
        [transforms.Resize((448, 448), interpolation=PIL.Image.BICUBIC),
         transforms.ToTensor(),
         normalize])

    train_dir = 'E:/Dataset/VG/VG_100K'
    train_anno = 'E:/Dataset/VG/vg200/train_list_500.txt'
    train_label = 'E:/Dataset/VG/vg200/vg_category_200_labels_index.json'
    test_dir = 'E:/Dataset/VG/VG_100K'
    test_anno = 'E:/Dataset/VG/vg200/test_list_500.txt'
    test_label = 'E:/Dataset/VG/vg200/vg_category_200_labels_index.json'
    train_dataset = VG('train', args,'train', train_dir, train_anno, train_label,
                       transform=train_data_transform)
    val_dataset = VG('val', args,'val',
                     test_dir,
                     test_anno,
                     test_label,
                     transform=test_data_transform)
    return train_dataset, val_dataset


def compute_cooccurrence_matrix(labels):
    num_samples, num_labels = labels.shape
    cooccurrence_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for label_vector in labels:
        for i in range(num_labels):
            if label_vector[i] == 1:
                for j in range(num_labels):
                    if label_vector[j] == 1:
                        cooccurrence_matrix[i][j] += 1
    return cooccurrence_matrix