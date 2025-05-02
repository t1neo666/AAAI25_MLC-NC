import os
import random
import xml.etree.ElementTree as ET
from sklearn.utils import shuffle
import numpy as np
from torchvision.transforms import transforms
from .get_voc import VOC2007_handler


# Image statistics
RGB_statistics = {
    'default': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    }
}
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

def parse_voc_annotation(ann_dir, img_dir, labels):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(os.path.join(ann_dir, ann))
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if labels and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

        if img['object']:
            all_imgs += [img]

    return all_imgs, seen_labels


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


def uniformly_sample_voc(samples, labels, num_samples=1142):
    total_samples = len(samples)
    interval = total_samples // num_samples

    sampled_indices = [i for i in range(0, total_samples, interval)]

    if len(sampled_indices) > num_samples:
        sampled_indices = sampled_indices[:num_samples]
    elif len(sampled_indices) < num_samples:
        sampled_indices.extend(random.sample(sampled_indices, num_samples - len(sampled_indices)))

    sampled_samples = [samples[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]

    return sampled_samples, sampled_labels


def get_uniform_voc(args):
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

    # 加载VOC数据集
    train_data_path = 'E:/Dataset/VOC2012'
    test_data_path = 'E:/Dataset/VOCtest'
    train_images, train_labels, test_images, test_labels = get_VOC2007(train_data_path, test_data_path)


    # 随机打乱数据
    all_images, all_labels = shuffle(train_images, train_labels, random_state=42)

    # 从数据集中均匀采样1142个样本
    sampled_images, sampled_labels = uniformly_sample_voc(all_images, all_labels)

    train_dataset = VOC2007_handler(sampled_images, sampled_labels, train_data_path, transform=train_transform)
    test_dataset = VOC2007_handler(test_images, test_labels, test_data_path, transform=test_transform)

    return train_dataset, test_dataset
