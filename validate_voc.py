import os
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from src_files.data.get_dataset import get_dataset
from src_files.helper_functions.helper_functions import mAP, AverageMeter
from src_files.models import create_model

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch MS_COCO validation')
parser.add_argument('--classifier', default='ETF', choices=['ETF', 'GroupFC'])
parser.add_argument('--data', type=str, default='E:/Dataset/voc2007')
parser.add_argument('--flag', default=11, type=int)
parser.add_argument('--dataname', help='dataname', default='voc2007', choices=['coco17', 'voc2007', 'voc2012'])
parser.add_argument('--model-name', default='resnet50')
parser.add_argument('--model-path', default='./mfm_models/voc2007_211.ckpt', type=str)
parser.add_argument('--num-classes', default=20)
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--thr', default=0.75, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=32, type=int,
                    metavar='N', help='print frequency (default: 64)')

parser.add_argument('--alpha', default=1, type=float)

# ML-Decoder
# parser.add_argument('--use-ml-decoder', default=1, type=int)
# parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
# parser.add_argument('--decoder-embedding', default=768, type=int)
# parser.add_argument('--zsl', default=0, type=int)

def draw_box(head_positive, head_negitive, tail_positive, tail_negitive):
    # 为每个数组创建对应的标记
    tail_negitive=np.array(tail_negitive)
    tail_negitive = tail_negitive[tail_negitive < 0.3]
    labels_head_positive = np.ones_like(tail_positive)  # 用1表示来自数组1
    labels_head_negative = np.zeros_like(tail_negitive)  # 用0表示来自数组2

    # 根据标记拼接数组
    head = np.concatenate([tail_positive, tail_negitive])
    labels = np.concatenate([labels_head_positive, labels_head_negative])
    # 创建 DataFrame，并给每一列添加标题
    df = pd.DataFrame({
        'logits': head,
        'class': labels
    })
    df['class'] = df['class'].replace({1: 'positive'})
    df['class'] = df['class'].replace({0: 'negative'})

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # 创建多个箱型图
    plt.figure(figsize=(8, 6))  # 可选，设置图形大小
    sns.boxplot(x="class", y="logits", data=df, hue="class",showfliers=False,palette='CMRmap')

    # 添加标题和标签
    plt.title("Predictios of tail classes")
    plt.xlabel("classes")
    plt.ylabel("predictions")
    # 移动图例到右上角
    plt.legend(loc='upper right')

    # 显示
    # 保存图形为EPS格式
    plt.savefig('./tail_boxplot.eps', format='eps')

    # 显示图形
    plt.show()

def main():
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print(model)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()
    ########### eliminate BN for faster inference ###########
    # model = model.cpu()
    # model = InplacABN_to_ABN(model)
    # model = fuse_bn_recursively(model)
    # model = model.cuda().half().eval()
    #######################################################
    print('done')

    train_dataset, val_dataset = get_dataset(args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    validate_multi(val_loader, model, args)


def validate_multi(val_loader, model, args):
    print("starting actuall validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    mAP_meter = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    head_positive = []
    head_negitive = []
    tail_positive = []
    tail_negitive = []
    for i, (input, target) in enumerate(val_loader):
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        for batch in range(len(target)):
            for i in range(len(target[0])):
                if int(target[batch][i].cpu()) == 0 and i in [14, 8, 6, 10, 4, 15]:
                    head_negitive.append(output[batch][i].cpu())
                elif int(target[batch][i].cpu()) == 1 and i in [14, 8, 6, 10, 4, 15]:
                    head_positive.append(output[batch][i].cpu())
                elif int(target[batch][i].cpu()) == 0 and i in [9, 16, 0, 18, 2, 12, 3, 5]:
                    tail_negitive.append(output[batch][i].cpu())
                elif int(target[batch][i].cpu()) == 1 and i in [9, 16, 0, 18, 2, 12, 3, 5]:
                    tail_positive.append(output[batch][i].cpu())
                else:
                    pass

        # measure accuracy and record loss
        pred = output.data.gt(args.thr).long()



    # mAP_score = mAP_scoreAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    # print("mAP score:", mAP_score)
    draw_box(head_positive, head_negitive, tail_positive, tail_negitive)

    return


if __name__ == '__main__':
    main()
