import os

import pandas as pd
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from randaugment import RandAugment

from src_files.data.extra_aug import PhotoMetricDistortion, RandomCrop
from src_files.helper_functions.helper_functions import CutoutPIL

plt.rcParams["font.sans-serif"] = "SimHei" #解决中文乱码问题

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

# Image statistics
RGB_statistics = {
    'default': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    }
}

class VOC2007_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' + self.X[index] + '.jpg').convert('RGB')

        scale = np.random.rand() * 2 + 0.25
        w = int(x.size[0] * scale)
        h = int(x.size[1] * scale)

        if min(w, h) < 227:
            scale = 227 / min(w, h)
            w = int(x.size[0] * scale)
            h = int(x.size[1] * scale)

        if self.random_crops == 0:
            x = self.transform(x)

        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]

        return x, y

    def __len__(self):
        return len(self.X)


def __dataset_info(data_path, trainval):
    classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, range(num_classes)))

    with open(data_path + '/ImageSets/Main/' + trainval + '.txt') as f:
        annotations = f.readlines()

    annotations = [n[:-1] for n in annotations]
    names = []
    labels = []
    for af in annotations:
        filename = os.path.join(data_path, 'Annotations', af)
        tree = ET.parse(filename + '.xml')
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes_cl = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            boxes_cl[ix] = cls

        lbl = np.zeros(num_classes)
        lbl[boxes_cl] = 1
        labels.append(lbl)
        names.append(af)

    labels = np.array(labels).astype(np.float32)
    labels = labels[:, 1:]

    return np.array(names), np.array(labels).astype(np.float32)


def get_VOC2007(train_data_path, test_data_path):
    train_data, train_labels = __dataset_info(train_data_path, 'trainval')
    train_idx = np.arange(train_labels.shape[0])
    # np.random.shuffle(train_idx)
    train_data, train_labels = train_data[train_idx], train_labels[train_idx]

    test_data, test_labels = __dataset_info(test_data_path, 'test')
    test_idx = np.arange(test_labels.shape[0])
    np.random.shuffle(test_idx)
    test_data, test_labels = test_data[test_idx], test_labels[test_idx]

    return train_data, train_labels, test_data, test_labels

def get_longtail():
    longtail = np.load('././appendix/VOCdevkit/longtail2012/class_freq.pkl',allow_pickle=True)
    longtail_labels = longtail['gt_labels']
    longtail_images = []
    with open('././appendix/VOCdevkit/longtail2012/img_id.txt', 'r') as f:
        data = f.readlines()
    for i, line in enumerate(data):
        longtail_images.append(line[:11])

    longtail_images = np.array(longtail_images)
    return longtail_images, longtail_labels

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
    np.save('../loss_functions/voc_co_occurrence.npy', corrlation_class)
    # torch.save(corrlation, 'E:/code/ASL-final/features/voc2007/corrlation_matrix.pth')


def count_class_labels(labels):
    dict_num = [0] * len(labels[0])
    for label in labels:
        indices = np.where(label == 1)[0]
        for index in indices:
            dict_num[index] += 1
    print(dict_num)
    calculate_corrlation(20,labels)
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
    plt.savefig('./im_voc.eps', format='eps')

    # 显示图形
    plt.show()
    print('save success')

def voc(args):
    rgb_mean, rgb_std = RGB_statistics['default']['mean'], RGB_statistics['default']['std']
    train_transform=None
    test_transform=None
    if args.flag == 11:
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

    if args.dataname == 'voc2007':
        train_data_path = 'E:/Dataset/VOC2012'
        test_data_path = 'E:/Dataset/VOCtest'
    else:
        print("can not find this dataset!")

    train_images, train_labels, test_images, test_labels = get_VOC2007(train_data_path, test_data_path)

    longtail_images, longtail_labels = get_longtail()
    train_dataset = VOC2007_handler(longtail_images, longtail_labels, train_data_path, transform=train_transform)
    test_dataset = VOC2007_handler(test_images, test_labels, test_data_path, transform=test_transform)

    return train_dataset, test_dataset


def count_head_tail_cooccurrence(longtail_labels):
    class_split = np.load(
        '///BCaL222//appendix//VOCdevkit//longtail2012//class_split.pkl',
        allow_pickle=True)
    head = list(class_split['head'])
    medium = list(class_split['middle'])
    tail = list(class_split['tail'])
    total_tail_count = 0
    head_and_tail_count = 0

    # Iterate through all the lt_labels
    for lt_label in longtail_labels:
        # Check if this label contains any tail label
        contains_tail = any(lt_label[i] == 1 for i in tail)

        if contains_tail:
            total_tail_count += 1
            # Check if this label also contains any head label
            contains_head_and_tail = any(lt_label[i] == 1 for i in head)
            if contains_head_and_tail:
                head_and_tail_count += 1

    # Calculate the proportion of lt_labels containing both head and tail labels
    if total_tail_count > 0:
        proportion = head_and_tail_count / total_tail_count
    else:
        proportion = 0
    # Proportion of lt_labels containing both head and tail labels
    return proportion