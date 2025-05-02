import numpy as np
import torch
import torch.nn as nn
import os
class FocalLoss(nn.Module):
    def __init__(self, weight, map_gamma=0.1, map_alpha=0.1, map_beta=10, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, distribution_path=None, co_occurrence_matrix=None):
        super(FocalLoss, self).__init__()

        self.weight = weight
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        co_occurrence_matrix = torch.tensor(np.load(
            '///BCaL222//src_files//loss_functions//voc_co_occurrence.npy')).cuda()
        freq_file = '///BCaL222//appendix//VOCdevkit//longtail2012//class_freq.pkl'
        self.class_freq = torch.from_numpy(np.load(freq_file, allow_pickle=True)['class_freq']).cuda()
        self.co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=0)
        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.map_gamma = map_gamma
        self.map_alpha = map_alpha
        self.map_beta = map_beta

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        y = pos_weight.cpu().data.numpy()
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        x = weight.cpu().data.numpy()
        return weight

    def forward(self, x, y, epoch):

        attention_scores_total = []
        for k in range(y.shape[0]):
            attention_scores = self.co_occurrence_matrix[y[k] == 1].mean(dim=0)
            attention_scores = attention_scores / attention_scores.sum()
            attention_scores_total.append(attention_scores)
        final_attention_scores = torch.stack(attention_scores_total, 0)

        #positive
        x_sigmoid = torch.pow(torch.sigmoid(x), 1)
        gamma_class_pos = self.gamma_class_pos - final_attention_scores
        # gamma_class_pos=1
        xs_pos = x_sigmoid * gamma_class_pos
        xs_neg = 1 - x_sigmoid

        #negative
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        xs_neg = torch.where(final_attention_scores == 0, xs_neg, xs_neg * self.gamma_class_ng).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        loss *= self.weight
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = self.weight * y + self.weight * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
        loss *= one_sided_w
        # if epoch > 11:
        # weight = self.rebalance_weight(y.float()).float()
        # loss *= weight
        return -loss.sum()