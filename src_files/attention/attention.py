import torch
from torch import nn
from .Proto_Classifiers import Proto_Classifier
import torch.nn.functional as F

def add_attention(args, model):
    num_classes = model.num_classes
    model.global_pool = nn.Identity()
    del model.fc
    model.fc = Attention(num_classes=num_classes, alpha=args.alpha, classifier=args.classifier)

    return model

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h, duplicate_pooling, out_extrap):
        for i in range(h.shape[1]):
            # h [64, 80, 768], h_i [64, 768], duplicate_pooling[num_class, feature, 1]
            h_i = h[:, i, :]
            w_i = duplicate_pooling[i, :, :]

            # out_extrap最终是[64, 80, 1]，torch.matmul是[64, 768]和[768, 1]的矩阵相乘
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)

class GroupProto:
    def __init__(self):
        pass

    def matmul_group(self, h, duplicate_proto, class_num, proj_dim):
        # Initialize the result tensor with the appropriate shape
        result = torch.zeros((h.shape[0], class_num, proj_dim), device=h.device, dtype=h.dtype)

        # Perform the matrix multiplication for each row of A and the corresponding slice of B
        for b in range(h.shape[0]):
            for i in range(class_num):
                result[b, i, :] = torch.matmul(h[b, i, :], duplicate_proto[i, :, :])

        return result
class Attention(nn.Module):
    def __init__(self, num_classes, alpha, classifier):
        super(Attention, self).__init__()
        embed_len_decoder = num_classes
        self.classifier = classifier
        self.alpha = alpha
        self.project_dimension = 20
        # switching to 768 initial embeddings
        self.embed_standart = nn.Linear(2048,768)
        self.sup_embed = nn.Linear(768, 300)
        # 80*768 嵌入向量
        self.query_embed_group = nn.Embedding(embed_len_decoder, 768)
        proto_query_embed = Proto_Classifier(768, num_classes).proto.T
        self.query_embed = proto_query_embed.detach().clone()
        self.temperature = nn.Parameter(torch.ones(1))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)

        # group fully-connected
        self.num_classes = num_classes
        # duplicate_pooling [80, 768, 1] 分类器权重
        self.duplicate_pooling = nn.Parameter(torch.Tensor(embed_len_decoder, 768, 1))
        self.duplicate_pooling_bias = nn.Parameter(torch.Tensor(num_classes))

        # 投影层
        self.duplicate_proto = nn.Parameter(torch.Tensor(num_classes, 768, self.project_dimension))
        self.duplicate_proto_bias = nn.Parameter(torch.Tensor(num_classes, self.project_dimension))
        self.projection_layers = nn.ModuleList([
            nn.Linear(768, self.project_dimension)
            for _ in range(num_classes)
        ])

        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)

        torch.nn.init.xavier_normal_(self.duplicate_proto)
        torch.nn.init.constant_(self.duplicate_proto_bias, 0)

        self.group_fc = GroupFC(embed_len_decoder)
        self.proto_fc = GroupProto()

        # ETF专用
        # 固定分类器
        self.proto_classifier = Proto_Classifier(num_classes*self.project_dimension, num_classes)
        self.rand_classifier = nn.Parameter(torch.Tensor(num_classes*self.project_dimension, num_classes))
        self.ablation_proto_classifier = Proto_Classifier(num_classes, num_classes)
        #self.ablation_proto_classifier = nn.Parameter(torch.Tensor(num_classes*self.project_dimension, num_classes))
        # 温度参数
        self.scaling_train = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7, 7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        # embedding_spatial [bs, 49, 2048]
        embedding_spatial_768 = self.embed_standart(embedding_spatial)
        # embedding_spatial_768 [bs, 49, 768]
        embedding_spatial_768 = torch.nn.functional.relu(embedding_spatial_768, inplace=False)

        bs = embedding_spatial_768.shape[0]

        # 语义向量
        if self.classifier == 'GroupFC':
            query_embed = self.query_embed_group.weight
        if self.classifier == 'ETF':
            query_embed = self.query_embed

        # 将query_embed 转化成 [num_classes, bs, 768]
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        # tgt: [num_class, 64(bs), 768]；embedding_spatial_768.transpose(0, 1) [49, 64, 768]

        h, attn_output_weights = self.multihead_attn(tgt, embedding_spatial_768.transpose(0, 1)
                                                               , embedding_spatial_768.transpose(0, 1))
        # h [64, 80, 768]
        h = h.transpose(0, 1)
        h = torch.where(h > 0, h * self.alpha, h)

        if self.classifier == 'GroupFC':
            h_proj = self.proto_fc.matmul_group(h, self.duplicate_proto, self.num_classes, self.project_dimension)
            h_proj += self.duplicate_proto_bias

            # 师姐的方法
            out_extrap = torch.zeros(h.shape[0], h.shape[1], 1, device=h_proj.device, dtype=h_proj.dtype)
            self.group_fc(h, self.duplicate_pooling, out_extrap)
            h_out = out_extrap.flatten(1)
            h_out += self.duplicate_pooling_bias
            logits = h_out
            feature_proj = h_proj

        # ETF分类器方法
        elif self.classifier == 'ETF':
            # ETF第一步 projection
            feature = self.proto_fc.matmul_group(h, self.duplicate_proto, self.num_classes, self.project_dimension)
            feature += self.duplicate_proto_bias
            # feature = [self.projection_layers[i](h[:, i, :]) for i in range(len(self.projection_layers))]
            # feature = torch.stack(feature, dim=1)
            # 计算每个特征的 L2 范数，保持维度以便后续广播操作
            l2_norm = torch.norm(feature, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            # 归一化特征
            feature_origin = torch.div(feature, l2_norm)
            feature = feature_origin.view(feature_origin.shape[0], -1)
            output_local = torch.matmul(feature, self.proto_classifier.proto)
            logits = output_local
            feature_proj = feature_origin
        # x = self.avgpool1(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # logits = logits + x

        #return logits, h, self.query_embed.weight, feature_proj, attn_output_weights
        return self.temperature*logits, h, self.query_embed, feature_proj


def compute_weight(tensorA, tensorB):
    tensorA = F.normalize(tensorA, p=2, dim=-1)
    tensorB = F.normalize(tensorB, p=2, dim=-1)
    tensorB_tran = torch.transpose(tensorB, -1, -2)
    bs = tensorB.shape[0]
    class_num = tensorA.shape[0]
    weight = torch.zeros(bs, class_num, tensorB.shape[1], device=tensorA.device)
    for i in range(bs):
        weight[i, :, :] = torch.matmul(tensorA, tensorB_tran[i, :, :])
    return weight
