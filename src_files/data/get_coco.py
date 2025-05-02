import json
import os
import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import torchvision.transforms as transforms
from randaugment import RandAugment
from PIL import Image
import torch
from torchvision import datasets as datasets

from src_files.helper_functions.helper_functions import CutoutPIL
# Image statistics
RGB_statistics = {
    'default': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    }
}
def calculate_corrlation(classes, labels):
    corrlation_class = np.zeros((classes, classes))
    corrlation = np.zeros((classes, classes))
    class_list = np.zeros((1, classes))  # 每个类出现在哪些图像index中，构造字典
    # 计算共现频次
    labels = np.array(labels)
    for i in range(labels.shape[0]):
        x = labels[i, :]
        pos_label = np.where(x == 1)[0]
        for k in pos_label:
            num = 0
            for j in pos_label:
                if k != j:
                    corrlation_class[k, j] += 1
                    num += 1
    np.save('./coco_co_occurrence.npy', corrlation_class)
    # torch.save(corrlation, 'E:/code/ASL-final/features/voc2007/corrlation_matrix.pth')

def count_class_labels(labels):
    dict_num = [0] * len(labels[0])
    for label in labels:
        indices = np.where(label == 1)[0]
        for index in indices:
            dict_num[index] += 1

    print(dict_num)
    calculate_corrlation(80, labels)

    # 使用enumerate给数组打标
    tagged_dict_num = [[i, value] for i, value in enumerate(dict_num)]
    # 将字典转为DataFrame
    columns = ['class', 'num']
    df = pd.DataFrame(tagged_dict_num, columns=columns)
    df = df.sort_values(by='num', ascending=False)

    plt.figure(figsize=(20, 6))  # 可选，设置图形大小
    # 使用Seaborn生成条形图
    sns.barplot(x='class', y='num', data=df)
    # 保存图形为EPS格式
    plt.savefig('./im_coco.eps', format='eps')

    # 显示图形
    plt.show()
    print('save success')

class CocoDetection_val(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class CocoDetection_train(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, longtail_images, longtail_labels, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(longtail_images)
        self.Y = longtail_labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_id = int(self.ids[index])
        target = self.Y[index]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CocoDetectionTrainWhole(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1

        target = output.max(dim=0)[0]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetectionTrainSampled(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, sample_size=1909, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        # 获取所有图像的ID并随机抽取指定数量的样本
        all_ids = list(self.coco.imgToAnns.keys())
        if sample_size < len(all_ids):
            self.ids = random.sample(all_ids, sample_size)
        else:
            self.ids = all_ids

        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1

        target = output.max(dim=0)[0]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)
def get_longtail():
    longtail = np.load('././appendix/coco/longtail2017/class_freq.pkl',allow_pickle=True)
    longtail_labels = longtail['gt_labels']
    longtail_images = []
    with open('././appendix/coco/longtail2017/img_id.pkl', 'r') as f:
        data = f.readlines()
    for i, line in enumerate(data):
        longtail_images.append(line[:-1])

    longtail_images = np.array(longtail_images)
    return longtail_images, longtail_labels

def get_coco(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    rgb_mean, rgb_std = RGB_statistics['default']['mean'], RGB_statistics['default']['std']
    # COCO Data loading
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)
    ])
    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)
    ])

    instances_path_val = os.path.join(args.data, 'annotations/instances_val2017.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2017.json')

    data_path_val = f'{args.data}/val2017'  # args.data
    data_path_train = f'{args.data}/train2017'  # args.data
    longtail_images, longtail_labels = get_longtail()

    train_dataset = CocoDetection_train(data_path_train, instances_path_train, longtail_images, longtail_labels, transform=train_transform)
    #train_dataset = CocoDetectionTrainWhole(data_path_train, instances_path_train, transform=train_transform)
    test_dataset = CocoDetection_val(data_path_val, instances_path_val, transform=test_transform)

    return train_dataset, test_dataset