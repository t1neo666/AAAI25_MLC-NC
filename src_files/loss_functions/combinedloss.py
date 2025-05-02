import torch.nn as nn
from .featurelabelloss import FeatureLabelLoss
from .twowayloss import TwoWayLoss
from .mfloss import MultiFocalLoss
from .prototypeloss import PrototypeLoss, ContrastiveProtoLoss
from .aslloss import AsymmetricLoss


class CombinedTwoWayLfaLoss(nn.Module):
    def __init__(self, weight_cosine=0.5, weight_twoway=0.5):
        super(CombinedTwoWayLfaLoss, self).__init__()
        self.cosine_loss = FeatureLabelLoss()
        self.twoway_loss = TwoWayLoss()
        self.weight_cosine = weight_cosine
        self.weight_twoway = weight_twoway

    def forward(self, logits, features, embeddings, labels):
        loss_cosine = self.cosine_loss(features, embeddings, labels)
        loss_twoway = self.twoway_loss(logits, labels)
        combined_loss = self.weight_cosine * loss_cosine + self.weight_twoway * loss_twoway
        return combined_loss


class CombinedMfLfaLoss(nn.Module):
    def __init__(self, args, gamma_neg=3, gamma_pos=1, gamma_class_ng=1.2, clip=0.05, disable_torch_grad_focal_loss=False, weight_cosine=0.5, weight_focal=0.5):
        super(CombinedMfLfaLoss, self).__init__()
        self.multi_focal_loss = MultiFocalLoss(args, gamma_neg, gamma_pos, gamma_class_ng, clip, disable_torch_grad_focal_loss)
        self.cosine_loss = FeatureLabelLoss()
        self.weight_cosine = weight_cosine
        self.weight_focal = weight_focal

    def forward(self, logits, features, embeddings, labels):
        # 计算 MultiFocalLoss
        loss_focal = self.multi_focal_loss(logits, labels)
        # 计算 FeatureLabelLoss
        loss_cosine = self.cosine_loss(features, embeddings, labels)
        # 合并两个损失，这里简单地做加权平均
        combined_loss = self.weight_cosine * loss_cosine + self.weight_focal * loss_focal
        return combined_loss


class CombinedTwoWayProtoLoss(nn.Module):
    def __init__(self, weight_prototype=0.5, weight_twoway=0.5):
        super(CombinedTwoWayProtoLoss, self).__init__()
        self.prototype_loss = PrototypeLoss()
        self.twoway_loss = TwoWayLoss()
        self.weight_prototype = weight_prototype
        self.weight_twoway = weight_twoway

    def forward(self, logits, class_prototype, features, labels):
        loss_proto = self.prototype_loss(class_prototype, features, labels)
        loss_twoway = self.twoway_loss(logits, labels)
        combined_loss = self.weight_prototype * loss_proto + self.weight_twoway * loss_twoway
        return combined_loss

class CombinedTwoWayProtoFLfaLoss(nn.Module):
    def __init__(self, weight_prototype=0.5, weight_twoway=0.5, weight_fla=0.5):
        super(CombinedTwoWayProtoFLfaLoss, self).__init__()
        self.prototype_loss = ContrastiveProtoLoss()
        self.twoway_loss = TwoWayLoss()
        self.fla_loss = FeatureLabelLoss()
        self.weight_prototype = weight_prototype
        self.weight_twoway = weight_twoway
        self.weight_fla = weight_fla

    def forward(self, logits, embeddings, class_prototype, features, feature_proj, labels):
        loss_proto = self.prototype_loss(class_prototype, feature_proj, labels)
        loss_twoway = self.twoway_loss(logits, labels)
        loss_fla = self.fla_loss(features, embeddings, labels)
        combined_loss = self.weight_prototype * loss_proto + self.weight_twoway * loss_twoway + self.weight_fla * loss_fla
        return combined_loss
