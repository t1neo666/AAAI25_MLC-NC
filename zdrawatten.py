import json
import cv2
import os
import xml.etree.ElementTree as ET
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src_files.data.get_voc import __dataset_info
from src_files.helper_functions.helper_functions import mAP, AverageMeter
from src_files.models import create_model
matplotlib.use('TkAgg')


parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')
parser.add_argument('--classifier', default='ETF', choices=['ETF', 'GroupFC'])
parser.add_argument('--data', type=str, default='E:/Dataset/voc2007')
parser.add_argument('--flag', default=11, type=int)
parser.add_argument('--model-path', default='./mfm_models/voc2007_ETF_alpha1epoch10.ckpt', type=str)
parser.add_argument('--pic_name', type=str, default='000002.jpg')
parser.add_argument('--pic_path', type=str, default='E:/Dataset/VOCtest/JPEGImages/000002.jpg')
parser.add_argument('--model-name', default='resnet50')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--dataset_type', type=str, default='VOC')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('--decoder-embedding', default=768
                    , type=int)
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=32, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--alpha', default=1, type=float)


name = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
def visulize_spatial_attention(img_path, attention_mask, ratio=1, cmap="jet", classes=0):
    """
    attention_mask: 2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:   attention style, default: "jet"
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    # scale表示放大或者缩小图片的比率
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    # plt.savefig('E:/pic/demo6/test_'+str(classes)+'.png', format="png")
    plt.show()


def draw_CAM(args, model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    im_resize = img.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()

    # 获取模型输出的feature/score
    model.eval()
    output, features, embeddings, feature_proj, weight_att = model(tensor_batch)

    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)[0]

    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

def main():
    print('ASL Example Inference code on a single image')

    # parsing args
    args = parser.parse_args()

    # setup model
    model = create_model(args).cuda()
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

    test_data, test_labels = __dataset_info('E:/Dataset/VOCtest', 'test')
    id = 0
    for i, x in enumerate(test_data):
        xjpg = f"{x}.jpg"
        if xjpg == args.pic_name:
            id = i
            break
    y = test_labels[id]
    y = y.astype(int)
    selected_names = [j for i, j in enumerate(name) if y[i] == 1]
    print(selected_names)
    # doing inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()

    # output,feature,weights = model(tensor_batch)
    output, features, embeddings, feature_proj, weight_att = model(tensor_batch)

    weights = weight_att[0].cpu().data.numpy()
    for index in range(20):
        if y[index] == 1:
            weight = weights[index][:]
            weight = weight.reshape((7, 7))
            visulize_spatial_attention(img_path=args.pic_path, attention_mask=weight)

    draw_CAM(args, model, args.pic_path, 'E:\\test.png')

    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    plt.rcParams["axes.titlesize"] = 10

    plt.show()
    print('done\n')

if __name__ == '__main__':
    main()
