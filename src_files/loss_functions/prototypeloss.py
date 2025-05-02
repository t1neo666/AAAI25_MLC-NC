import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrototypeLoss(nn.Module):
    def __init__(self):
        super(PrototypeLoss, self).__init__()
        # class_split = np.load('appendix/VOCdevkit/longtail2012/class_split.pkl', allow_pickle=True)
        class_split = np.load('appendix/coco/longtail2017/class_split.pkl', allow_pickle=True)
        self.head = list(class_split['head'])

    def forward(self, class_prototype, feature_proj, labels):
        # class_prototype: [num_classes, 768]
        # features: [batch_size, num_classes, 768]
        # labels: [batch_size, num_classes]

        loss = 0.0
        count = 0  # 计数有效的类原型使用次数

        # 遍历每个样本和每个类
        for i in range(labels.shape[0]):  # 遍历每个样本
            for j in range(labels.shape[1]):  # 遍历每个类
                if labels[i, j] == 1:  # and j not in self.head:  # 如果样本i中存在类j并且j在head列表中
                    # 提取属于当前类别的特征向量和对应的类原型
                    feature_vec = feature_proj[i, j]
                    proto_vec = class_prototype[j]

                    # 计算二范数损失
                    norm_loss = torch.norm(feature_vec - proto_vec)

                    # 累加损失
                    loss += norm_loss ** 2
                    count += 1

        if count > 0:
            loss = loss / count
        else:
            loss = torch.tensor(0.0, device=feature_proj.device)  # 避免除以零

        return loss


class ContrastiveProtoLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveProtoLoss, self).__init__()
        self.temperature = temperature

    def forward(self, class_prototype, feature_proj, labels):

        batchsize, class_num, feature_dim = feature_proj.shape

        # Normalize class prototypes and feature projections
        class_prototype = F.normalize(class_prototype, dim=1)
        feature_proj = F.normalize(feature_proj, dim=2)

        losses = []

        for i in range(batchsize):
            for j in range(class_num):
                if labels[i, j] == 1:
                    # Compute all similarities (denominator of softmax)
                    all_sims = []
                    for k in range(class_num):
                        sim = torch.matmul(feature_proj[i, j, :], class_prototype[k, :]) / self.temperature
                        all_sims.append(sim)

                    all_sims = torch.stack(all_sims)

                    # Compute softmax loss for this (i, j) pair
                    log_prob = F.log_softmax(all_sims, dim=0)
                    loss = -log_prob[j]
                    losses.append(loss)

        # Compute mean loss over all positive pairs
        if losses:
            total_loss = torch.mean(torch.stack(losses))
        else:
            total_loss = torch.tensor(0.0)

        return total_loss


if __name__ == '__main__':
    class_prototype = torch.randn(10, 80)  # Assuming 10 classes
    feature_proj = torch.randn(32, 10, 80)  # Assuming batchsize of 32
    labels = torch.randint(0, 2, (32, 10))  # Binary labels for each class
    critrion = ContrastiveProtoLoss(temperature=0.5)
    loss = critrion(class_prototype, feature_proj, labels)
    print(loss)