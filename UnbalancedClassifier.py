from src_files.models import create_model
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='PyTorch tsne visualization')
parser.add_argument('--data', type=str, default='E:/Dataset/voc2007')
parser.add_argument('--classifier', default='GroupFC', choices=['ETF', 'GroupFC'])
parser.add_argument('--dataname', help='dataname', default='voc2007', choices=['coco17', 'voc2007', 'voc2012'])
parser.add_argument('--model-name', default='resnet50')
parser.add_argument('--model-path-biased', default='./mfm_models/voc2007_ETF_trainable.ckpt', type=str)
parser.add_argument('--model-path-unbiased', default='./mfm_models/voc2007_ETF_fixed.ckpt', type=str)
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


seed = 1

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
def main():
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    #print(model)
    state = torch.load(args.model_path_biased, map_location='cpu')
    model.load_state_dict(state, strict=False)
    # 删除为1的维度
    A = np.squeeze(state['fc.proto_classifier.proto'])

    state = torch.load(args.model_path_unbiased, map_location='cpu')
    model.load_state_dict(state, strict=False)
    B = np.squeeze(state['fc.proto_classifier.proto'])

    # 计算每个列向量的点积
    dot_products = torch.sum(A * B, dim=0)

    # 计算每个列向量的范数
    norms_A = torch.norm(A, dim=0)
    norms_B = torch.norm(B, dim=0)

    # 计算余弦相似度
    cosine_similarities = dot_products / (norms_A * norms_B)

    print(cosine_similarities)
    print(cosine_sim)

    # 计算与-1/19的MAE
    target_value = -1 / 19
    mae_matrix1 = torch.abs(cosine_similarity1 - target_value)
    mae_matrix2 = torch.abs(cosine_similarity2 - target_value)

    # 将结果转换为numpy数组以便于可视化
    mae_matrix_np1 = mae_matrix1.cpu().numpy()
    mae_matrix_np2 = mae_matrix2.cpu().numpy()

    # 创建图形和子图
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # 绘制第一个张量的热图
    sns.heatmap(mae_matrix_np1, annot=False, cmap='inferno', cbar=True, ax=axs[0], vmin=0, vmax=0.9)
    axs[0].set_title('MAE of Cosine Similarity and -1/19 (Tensor 1)')

    # 绘制第二个张量的热图
    sns.heatmap(mae_matrix_np2, annot=False, cmap='inferno', cbar=True, ax=axs[1], vmin=0, vmax=0.9)
    axs[1].set_title('MAE Cosine Similarity')

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(mae_matrix_np1, annot=False, cmap='inferno', cbar=True, ax=ax1, vmin=0, vmax=0.9)
    ax1.set_title('MAE Cosine Similarity')
    fig1.savefig('mae_heatmap_tensor1.pdf')

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(mae_matrix_np2, annot=False, cmap='inferno', cbar=True, ax=ax2, vmin=0, vmax=0.9)
    ax2.set_title('MAE Cosine Similarity')
    fig2.savefig('mae_heatmap_tensor2.pdf')

    plt.show()

if __name__ == '__main__':
    main()
