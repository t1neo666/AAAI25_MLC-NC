import numpy as np
import torch
import torch.nn as nn
import os
class DbLoss(nn.Module):
    def __init__(self):
        super(DbLoss, self).__init__()
        self.alpha = 0.1
        self.beta = 10
        self.mu = 0.2
        self.kappa = 0.05
        self.lamda = 5
        self.gamma = 2
        self.C = 200
        self.IsFocal = False
        C = 200

        freq_file = '///BCaL222//appendix//VOCdevkit//longtail2012//class_freq.pkl'
        class_freq = torch.from_numpy(np.load(freq_file, allow_pickle=True)['class_freq']).cuda()

        co_occurrence_matrix = 'C://Users//Y//Desktop//BCaL2222//BCaL222//src_files//loss_functions//voc_co_occurrence.npy'
        co_occurrence_matrix = torch.tensor(np.load(co_occurrence_matrix, allow_pickle=True)).cuda()

        P_C_i = 1 / (C * class_freq[:])
        co_occurrence_matrix_reverse = torch.where(co_occurrence_matrix == 0, torch.tensor(0.0, device='cuda:0'), 1 / co_occurrence_matrix)
        P_I = (1 / C) * co_occurrence_matrix_reverse.sum(axis=0)
        r_i_k = P_C_i / P_I
        self.r_hat = self.alpha + 1 / (1 + torch.exp(-self.beta * (r_i_k - self.mu)))
        pi = class_freq / class_freq.sum()
        bi_hat = -torch.log(1 / pi - 1)
        self.vi = -self.kappa * bi_hat

    def forward(self, x, y):
        batch_size, C = x.size()

        # 扩展 vi 和 r_hat 以匹配批量大小
        vi = self.vi.unsqueeze(0).expand(batch_size, -1)
        r_hat = self.r_hat.unsqueeze(0).expand(batch_size, -1)
        sigmoid_x = torch.sigmoid(x)
        if not self.IsFocal:
            # 计算第一个项
            term1 = y * torch.log(1 + torch.exp(-x + vi))
            # 计算第二个项
            term2 = (1 - y) * torch.log(1 + torch.exp(self.lamda * (x - vi))) / self.lamda
            # 计算损失
            loss = (1 / self.C) * torch.sum(r_hat * (term1 + term2), dim=1).mean()
        else:
            # 计算第一个项
            term1 = y * (1-sigmoid_x)**self.gamma * torch.log(1 + torch.exp(-x + vi))
            # 计算第二个项
            term2 = (1 - y) * sigmoid_x**self.gamma * torch.log(1 + torch.exp(self.lamda * (x - vi))) / self.lamda
            # 计算损失
            loss = (1 / self.C) * torch.sum(r_hat * (term1 + term2), dim=1).mean()
        return loss

def main():
    # 初始化损失函数
    loss_fn = DbLoss()

    # 生成假数据
    batch_size = 32
    num_classes = 200
    x = torch.randn(batch_size, num_classes).cuda()  # 模型输出的logits
    y = torch.randint(0, 2, (batch_size, num_classes)).float().cuda()  # 真实标签

    # 计算损失
    loss = loss_fn(x, y)
    print(f'计算的损失值: {loss.item()}')


if __name__ == '__main__':
    main()

