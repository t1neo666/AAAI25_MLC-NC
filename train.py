import os
import argparse
import random
import torch.nn as nn
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F
from collections import Counter
from torch.optim import lr_scheduler
import torchvision.models as models
from src_files.helper_functions.helper_functions import mAP, ModelEma, add_weight_decay, source_import
from src_files.helper_functions.logger import setup_logger
from src_files.loss_functions.dbloss import DbLoss
from src_files.loss_functions.aslloss import AsymmetricLoss
from src_files.loss_functions.bceloss import BCELoss
from src_files.loss_functions.mfloss import MultiFocalLoss
from src_files.loss_functions.twowayloss import TwoWayLoss
from src_files.loss_functions.combinedloss import CombinedTwoWayLfaLoss, CombinedMfLfaLoss, CombinedTwoWayProtoLoss, CombinedTwoWayProtoFLfaLoss
from src_files.loss_functions.featurelabelloss import FeatureLabelLoss
from src_files.loss_functions.prototypeloss import PrototypeLoss, ContrastiveProtoLoss
from src_files.models import create_model
from src_files.loss_functions.losses import FocalLoss
from torch.cuda.amp import GradScaler, autocast
from src_files.data.get_dataset import get_dataset
from sklearn.metrics import average_precision_score


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--flag', default=11, type=int)
parser.add_argument('--dataname', help='dataname', default='voc2007', choices=['coco17', 'voc2007', 'vg'])
parser.add_argument('--data', help='path to dataset', default='E:/Dataset/VOC2007')
parser.add_argument('--d', default=20, type=int, help='projection dimension')
parser.add_argument('--num-classes', default=20)
parser.add_argument('--classifier', default='ETF', choices=['ETF', 'GroupFC'])
parser.add_argument('--output', metavar='DIR', default='./output/abl/baseline+mfm/voc', help='path to output folder')
parser.add_argument('--epoch', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--model-name', default='resnet50')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size,default=32')
parser.add_argument('--loss', default='CombinedTwoWayProtoFLfaLoss',
                    choices=['MultiFocalLoss', 'BCELoss', 'AsymmetricLoss', 'TwoWayLoss', 'CombinedTwoWayLfaLoss',
                             'CombinedMfLfaLoss', 'ClassCollapseLoss', 'CombinedTwoWayProtoLoss',
                             'CombinedTwoWayProtoFLfaLoss', 'PrototypeLoss'])
parser.add_argument('--finetune', default=False, choices=[True, False])
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay',
                    help='default=-4')
parser.add_argument('--lr', default=1e-4, type=float)

parser.add_argument('--model-path', default='./resnet50.pth', type=str)
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')

parser.add_argument('--alpha', default=1, type=float)

#
# parser.add_argument('--weight', default=3, type=int)
# parser.add_argument('--clip', default=0.05, type=float)
# parser.add_argument('--map_gamma', default=0.15, type=float)
# parser.add_argument('--map_alpha', default=0.1, type=float)
# parser.add_argument('--map_beta', default=10, type=float)

# stage1
# parser.add_argument('--weight', default=3, type=int)
# parser.add_argument('--clip', default=0.05, type=float)
# parser.add_argument('--map_gamma', default=0, type=float)
# parser.add_argument('--map_alpha', default=0, type=float)
# parser.add_argument('--map_beta', default=0, type=float)

# mfmloss
# parser.add_argument('--gamma_neg', default=2, type=int)
# parser.add_argument('--gamma_pos', default=1, type=int)
# parser.add_argument('--clip', default=0.05, type=float)
# parser.add_argument('--distribution_path', default='./src_files/loss_functions/voc_distribution.txt')
# parser.add_argument('--path', default='./src_files/loss_functions/voc_co_occurrence.npy')

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    args = parser.parse_args()

    # Setup model
    print('creating model {} with classifier:{}, alpha:{}, Epoch:{}, loss:{}, finetune:{}'.format(args.model_name,
                                                                                     args.classifier, args.alpha,
                                                                                     args.epoch, args.loss, args.finetune))
    model = create_model(args).cuda()
    # model = models.resnet101(pretrained=True)

    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state.items() if
                         (k in model.state_dict() and 'fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)

    print('done')
    logger = setup_logger(output=args.output, color=False, name="ASL")
    logger.info('creating model {} with classifier:{}, alpha:{}, Epoch:{}, loss:{}, finetune:{}'
                .format(args.model_name, args.classifier, args.alpha, args.epoch, args.loss, args.finetune))

    train_dataset, val_dataset = get_dataset(args)

    # Pytorch Data loader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    sampler = {'def_file': './src_files/data/sampler.py', 'num_samples_cls': '4', 'type': 'ClassAwareSampler'}
    sampler_dic = {
        'sampler': source_import(sampler['def_file']).get_sampler(),
    }
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               sampler=sampler_dic['sampler'](train_dataset,
                                                                              num_classes=args.num_classes),
                                               shuffle=False, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(args, model, train_loader, val_loader, args.lr, logger)
    logger.info("\n \n \n")


def train_multi_label_coco(args, model, train_loader, val_loader, lr, logger):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = args.epoch
    weight_decay = args.weight_decay

    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                       pct_start=0.2)

    #model.fc.proto_classifier.proto.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    highest_mAP = 0
    highest_mAP_global = 0
    highest_epoch = 0
    highest_global_epoch = 0
    trainInfoList = []
    maelist = []
    class_feature_proj_sum = torch.zeros(args.num_classes, 20).cuda()
    class_proj_counts = torch.zeros(args.num_classes).cuda()
    class_proj_prototype = torch.zeros(args.num_classes, 20).cuda()
    scaler = GradScaler()

    for epoch in range(Epochs):
        # # 检查哪些参数需要梯度
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

        # 用来做mae角度的
        # class_feature_sum = torch.zeros(args.num_classes, 768).cuda()
        # class_counts = torch.zeros(args.num_classes).cuda()
        # class_prototype = torch.zeros(args.num_classes, 768).cuda()

        if epoch < 5:
            # warmup
            model.fc.proto_classifier.proto.requires_grad = True
            criterion = CombinedTwoWayProtoFLfaLoss(weight_prototype=0.0, weight_twoway=0, weight_fla=1)
        else:
            model.fc.proto_classifier.proto.requires_grad = True
            criterion = CombinedTwoWayProtoFLfaLoss(weight_prototype=0.1, weight_twoway=1, weight_fla=0.5)

        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            # y = target.cpu().data.numpy()

            with autocast():
                output, features, embeddings, feature_proj = model(inputData)
                output = output.float()

            if epoch < 5:
                loss = criterion(output, embeddings, class_proj_prototype, features, feature_proj, target)
            else:
                loss = criterion(output, embeddings, class_proj_prototype, features, feature_proj, target)

            # loss = criterion(output, target)

            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            ema.update(model)

            # #统计这个batch中的所有feature
            with torch.no_grad():
                mask = target.bool()  # 假设target是一个[batch_size, num_classes]的多标签矩阵
                for idx in range(args.num_classes):
                    class_mask = mask[:, idx]  # 为第idx类获取掩码
                    if class_mask.any():
                        class_feature_proj_sum[idx] += feature_proj[class_mask, idx].sum(dim=0)  # 累加当前类的特征
                        class_proj_counts[idx] += class_mask.sum()  # 累加当前类的计数

            # # 统计这个batch中的所有feature
            # with torch.no_grad():
            #     mask = target.bool()  # 假设target是一个[batch_size, num_classes]的多标签矩阵
            #     for idx in range(args.num_classes):
            #         class_mask = mask[:, idx]  # 为第idx类获取掩码
            #         if class_mask.any():
            #             class_feature_sum[idx] += features[class_mask, idx].sum(dim=0)  # 累加当前类的特征
            #             class_counts[idx] += class_mask.sum()  # 累加当前类的计数

            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                            .format(epoch, Epochs, i, len(train_loader), scheduler.get_last_lr()[0], loss.item()))

        model.eval()

        class_proj_counts = class_proj_counts.view(args.num_classes, 1)
        class_proj_prototype = class_feature_proj_sum / class_proj_counts

        # class_counts = class_counts.view(args.num_classes, 1)
        # class_prototype = class_feature_sum / class_counts

        # mae = compute_prototype_mae(class_prototype)
        # mae_current = mae
        # maelist.append(mae_current)
        # logger.info('maelist:{}'.format(maelist))

        mAP_score, mAP_global_score = validate_multi(args, val_loader, model, ema, logger)
        model.train()
        if mAP_score > highest_mAP or mAP_global_score > highest_mAP_global:
            if mAP_score > highest_mAP:
                highest_epoch = epoch
                highest_mAP = mAP_score
            if mAP_global_score > highest_mAP_global:
                highest_global_epoch = epoch
                highest_mAP_global = mAP_global_score
            torch.save(model.state_dict(),
                       os.path.join('mfm_models/', args.dataname + '_' + str(args.classifier) + '_'
                                    + "alpha" + str(args.alpha) + 'best' + '.ckpt'))

        logger.info('model {} with classifier:{}, alpha:{}, Epoch:{}, loss:{}, finetune:{}'
                    .format(args.model_name, args.classifier, args.alpha, args.epoch, args.loss, args.finetune))
        logger.info("{} | Current mAP {}, Current mAP_global {} in ep {}".format(epoch, mAP_score,mAP_global_score, epoch))
        logger.info("highest mAP {} in ep {}, highest mAP_global {} in ep {}".format(highest_mAP, highest_epoch, highest_mAP_global, highest_global_epoch))
        logger.info("------------------------------------------------------------------\n")

def validate_multi(args, val_loader, model, ema_model, logger):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        if args.dataname == 'coco17':
            target = target.max(dim=1)[0]
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())[0]).cpu()
                output_ema = Sig(ema_model.module(input.cuda())[0]).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())


    if args.dataname == 'voc2007':
        class_split = np.load('appendix/VOCdevkit/longtail2012/class_split.pkl', allow_pickle=True)
    if args.dataname == 'coco17':
        class_split = np.load('appendix/coco/longtail2017/class_split.pkl', allow_pickle=True)
    if args.dataname == 'vg':
        class_split = np.load('appendix/vg/class_split.pkl', allow_pickle=True)
    head = list(class_split['head'])
    medium = list(class_split['middle'])
    tail = list(class_split['tail'])
    mAP_score_regular, mAP_head, mAP_medium, mAP_tail, AP = mAP(torch.cat(targets).numpy(),
                                                                torch.cat(preds_regular).numpy(),
                                                                head, medium, tail)
    mAP_score_ema, mAP_head_ema, mAP_medium_ema, mAP_tail_ema, AP_ema = mAP(torch.cat(targets).numpy(),
                                                                            torch.cat(preds_ema).
                                                                            numpy(), head, medium, tail)
    mAP_score_global = calculate_global_map(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    logger.info("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    logger.info("regular head{:.2f}, medium{:.2f}, tail{:.2f}; EMA head{:.2f}, medium{:.2f}, tail{:.2f}".format(
        mAP_head, mAP_medium, mAP_tail, mAP_head_ema, mAP_medium_ema, mAP_tail_ema))
    logger.info("AP:{}".format(AP))
    return mAP_score_regular, mAP_score_global


def compute_prototype_mae(class_prototype):
    class_num = class_prototype.size(0)
    similarities = []

    # 计算两两之间的余弦相似度
    for i in range(class_num):
        for j in range(i + 1, class_num):
            similarity = F.cosine_similarity(class_prototype[i].unsqueeze(0), class_prototype[j].unsqueeze(0))
            similarities.append(similarity.item())

    # 转换为张量
    similarities = torch.tensor(similarities)

    # 计算 MAE 并减去 -1/19
    mae = torch.mean(torch.abs(similarities + (1 / 19)))

    return mae.item()

def calculate_map(y_true, y_scores):
    ap_scores = []
    for i in range(y_true.shape[1]):
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    return map_score*100


def calculate_global_map(y_true, y_scores):
    ap_scores = []
    for i in range(y_true.shape[0]):
        ap = average_precision_score(y_true[i, :], y_scores[i, :])
        ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    return map_score*100


if __name__ == '__main__':
    main()
