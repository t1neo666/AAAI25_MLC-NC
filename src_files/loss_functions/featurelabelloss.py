import torch
import torch.nn.functional as F
import torch.nn as nn

class FeatureLabelLoss(nn.Module):
    def __init__(self):
        super(FeatureLabelLoss, self).__init__()

    def forward(self, features, embeddings, labels):
        # 标准化 features 和 embeddings
        features_norm = F.normalize(features, p=2, dim=2).float()  # 归一化到单位向量
        embeddings_norm = F.normalize(embeddings, p=2, dim=1).float()  # 归一化到单位向量

        # 将 embeddings 从 [20, 768] 扩展为 [64, 20, 768]
        # 首先增加一个新的维度，然后扩展这个维度到 batch_size 大小
        embeddings_norm = embeddings_norm.unsqueeze(0).expand(features.size(0), -1, -1)  # 使用 expand 进行扩展

        # 计算余弦相似度
        # features_norm 的形状为 [64, 20, 768], embeddings_norm 的形状为 [64, 20, 768]
        # 使用 transpose 调整 embeddings_norm 的形状，以进行正确的矩阵乘法
        cosine_similarities = torch.bmm(features_norm, embeddings_norm.transpose(1, 2))  # 结果形状为 [64, 20, 20]

        # 从 cosine_similarities 中提取对角线上的相似度值
        similarities = torch.diagonal(cosine_similarities, dim1=-1, dim2=-2)  # 结果形状为 [64, 20]

        # 调整相似度
        C = embeddings.size(0)
        S_i_c = (1 + similarities) / 2 + 1e-6  # 防止 log(0)

        # 计算修改后的二元交叉熵损失
        pos_loss = labels * torch.log(S_i_c)
        neg_loss = (1 - labels) * torch.log(1 - (C - 1) / C * torch.abs(1 / (C - 1) + similarities) + 1e-6)

        loss = -(pos_loss + neg_loss).mean()

        return loss

# 示例使用
if __name__ == '__main__':
    batch_size, num_classes, feature_dim = 64, 20, 768
    features = torch.randn(batch_size, num_classes, feature_dim)
    embeddings = torch.randn(num_classes, feature_dim)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()

    loss_func = FeatureLabelLoss()
    loss = loss_func(features, embeddings, labels)
    print("计算出的损失 Loss:", loss.item())
