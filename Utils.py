import argparse
import math
import os
import errno
import os.path as osp
import sys
import random

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.nn import Parameter, Module

import image_utils
from Model_pre import Backbone_onlyGlobal, SANN_select, SANN_fus
from Model import *
import concat_feature_1024
import Exp_Conv
from AdversarialNetwork import AdversarialNetwork
import seaborn as sns
from mindspore.nn.metrics import CosineSimilarity



class AverageMeter(object):
    '''Computes and stores the sum, count and average'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val
        self.count += count

        if self.count == 0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count


def str2bool(input):
    if isinstance(input, bool):
        return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()  # [64, 7]
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0], )
    target = target.astype(np.int32).reshape(target.shape[0], )

    for i in range(7):
        TP = np.sum((pred == i) * (target == i))
        TN = np.sum((pred != i) * (target != i))

        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred == target), pred.shape[0])

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP, np.sum(pred == i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP, np.sum(target == i))


def get_wrong_image(args, pred, target):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()  # [64, 7]
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0], )
    target = target.astype(np.int32).reshape(target.shape[0], )
    for i in range(len(pred)):
        print(pred[i] == target[i])


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix


# def print_confusion_matrix(conf_matrix):
#     # 绘制混淆矩阵
#     Emotion_kinds = 7  # 这个数值是具体的分类数，大家可以自行修改
#     labels = ["Surprise", 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']  # 每种类别的标签
#     conf_matrix = conf_matrix / torch.sum(conf_matrix, dim=1)
#     f, ax = plt.subplots()
#
#     # 显示数据
#     # plt.imshow(conf_matrix, cmap=plt.cm.Blues)
#
#     # # 在图中标注数量/概率信息
#     thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
#     for x in range(Emotion_kinds):
#         for y in range(Emotion_kinds):
#             # 注意这里的matrix[y, x]不是matrix[x, y]
#             info = int(conf_matrix[y, x])
#             plt.text(x, y, info,
#                      verticalalignment='center',
#                      horizontalalignment='center',
#                      color="white" if info > thresh else "black")
#
#     # df_cm = pd.DataFrame(conf_matrix, labels, labels)
#     # sns.heatmap(df_cm, annot=True, fmt=".2f", cmap='Blues', cbar=False, ax=ax)
#     #
#     plt.rc('font', family='Times New Roman')
#     plt.yticks(range(Emotion_kinds), labels)
#     plt.xticks(range(Emotion_kinds), labels, rotation=45)  # X轴字体倾斜45°
#     # label_y = ax.get_yticklabels()
#     # plt.setp(label_y, horizontalalignment='right')
#     # label_x = ax.get_xticklabels()
#     # plt.setp(label_x, rotation=45, horizontalalignment='center')
#     plt.tight_layout()  # 保证图不重叠
#     plt.savefig(fname='confusion_matrix.png', format="png")

def print_confusion_matrix(conf_matrix):
    """draw confusion matrix

    Args:
        data (dict): Contain config data
        path (path-like): The path to save picture
    """
    conf_matrix = conf_matrix / torch.sum(conf_matrix, dim=1, keepdim=True)
    # draw
    plt.figure()
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)  # 可以改变颜色
    labels = ["SU", 'FE', 'DI', 'HA', 'SA', 'AN', 'NE']  # 每种类别的标签
    indices = list(range(7))
    plt.xticks(indices, labels, rotation=45)
    plt.yticks(indices, labels)
    # plt.xlabel('pred')
    # plt.ylabel('true')
    # 显示数据
    for first_index in range(7):  # trues
        for second_index in range(7):  # preds
            if conf_matrix[second_index][first_index] < 0.55:
                plt.text(first_index, second_index, "{:.2f}".format(conf_matrix[second_index][first_index].item()),
                         verticalalignment='center', horizontalalignment='center')
            else:
                plt.text(first_index, second_index, "{:.2f}".format(conf_matrix[second_index][first_index].item()),
                         verticalalignment='center', horizontalalignment='center', color='white')

    plt.tight_layout()
    plt.savefig(fname='confusion_matrix.png', format="png")
    plt.close()


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def BulidModel(args):
    """Bulid Model."""

    if args.Backbone == 'ResNet18':
        if args.Network == 'Baseline':
            model = Baseline(args, 18, num_classes=7)
        elif args.Network == 'AConv':
            model = AConv(args, 18, num_classes=7)
        elif args.Network == 'DHSA':
            model = DHSA(args, 18, num_classes=7)
        elif args.Network == 'AConv+DHSA':
            model = AConv_DHSA(args, 18, num_classes=7)
        elif args.Network == 'AConv+DHSA+global':
            model = AConv_DHSA_global(args, 18, num_classes=7)
        elif args.Network == 'AConv+DHSA+global(x)':
            model = Exp_Conv.AConv_DHSA_global_x(args, 18, num_classes=7)
        elif args.Network == 'unified':
            model = unified(args, 18, num_classes=7)
        elif args.Network == 'test11':
            model = Exp_Conv.AConv_DHSA_global_x_11(args, 18, num_classes=7)
        elif args.Network == 'test21':
            model = Exp_Conv.AConv_DHSA_global_x_21(args, 18, num_classes=7)
        elif args.Network == 'test_sk':
            model = Exp_Conv.AConv_DHSA_global_x_SK(args, 18, num_classes=7)
        elif args.Network == 'test22':
            model = Exp_Conv.AConv_DHSA_global_x_22(args, 18, num_classes=7)
    elif args.Backbone == 'ResNet50':
        model = SANN_select(args, 50, num_classes=7)

    if args.Resume_Model != 'None':
        print('Resume Model: {}'.format(args.Resume_Model))
        checkpoint = torch.load(args.Resume_Model, map_location='cpu')

        model.load_state_dict(checkpoint, strict=False)
    else:
        print('No Resume Model')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    return model


def BulidAdversarialNetwork(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    ad_net = AdversarialNetwork(model_output_num, 128)
    # ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return ad_net


def DANN(features, ad_net):
    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    ad_out = ad_net(features)
    batch_size = ad_out.size(0)
    dc_target = torch.from_numpy(np.array([[1]] * (batch_size // 2) + [[0]] * (batch_size // 2))).float()

    ad_out = ad_out.cuda()
    dc_target = dc_target.cuda()

    return nn.BCELoss()(ad_out, dc_target)


def Show_Accuracy(acc, prec, recall, class_num=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg == 0 or recall[i].avg == 0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2 * prec[i].avg * recall[i].avg / (prec[i].avg + recall[i].avg)

    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg += acc[i].avg
        prec_avg += prec[i].avg
        recall_avg += recall[i].avg
        f1_avg += f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg / class_num, prec_avg / class_num, recall_avg / class_num, f1_avg / class_num

    # Logs Accuracy Infomation
    Accuracy_Info = ''

    Accuracy_Info += '    Accuracy'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(acc[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    Precision'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(prec[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    Recall'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(recall[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    F1'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(f1[i].avg)
    Accuracy_Info += '\n'

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg


def Compute_Similarity(feature, label, inter, intra):
    loss_inter, loss_intra = 0, 0
    bs = label.shape[0]
    feature = F.normalize(feature, dim=1)
    matrix = torch.mm(feature, feature.t())
    label = label.unsqueeze(dim=1)
    label_t = label.t()
    label = label.repeat(1, bs)
    label_t = label_t.repeat(bs, 1)
    for i in range(7):
        index_inter = (label == i).type(torch.uint8)
        index_inter_t = (label_t == i).type(torch.uint8)
        mask_inter = index_inter * index_inter_t
        mask_inter = torch.triu(mask_inter, diagonal=1)
        inter_num = torch.sum(mask_inter)
        inter_sim = matrix * mask_inter
        inter_sim = torch.sum(inter_sim)
        inter[i].update(inter_sim, inter_num)
        loss_inter = loss_inter + inter_sim / (inter_num + 1e-6)

        index_intra = (label != i).type(torch.uint8)
        index_intra_t = (label_t != i).type(torch.uint8)
        mask_intra = index_intra * index_intra_t
        mask_intra = torch.triu(mask_intra, diagonal=1)
        intra_num = torch.sum(mask_intra)
        intra_sim = matrix * mask_intra
        intra_sim = torch.sum(intra_sim)
        intra[i].update(intra_sim, intra_num)
        loss_intra = loss_intra + intra_sim / (intra_num + 1e-6)

    return loss_inter, loss_intra


def Similarity_Loss(args, feature, label, inter, intra):
    loss_inter, loss_intra = Compute_Similarity(feature, label, inter, intra)
    loss_inter, loss_intra = loss_inter / 7, loss_intra / 7

    return -loss_inter, loss_intra


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=7, feat_dim=1024, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))  # return centers,一行是一个中心

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class MMD_loss(nn.Module):
    def __init__(self):
        super(MMD_loss, self).__init__()

    def forward(self, source, target):
        delta = source - target
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss


class mkMMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(mkMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (224, 224)
    bz, c, h, w = feature_conv.shape
    output_cam = []
    for batch_size in range(bz):
        cam = weight_softmax[class_idx[0][batch_size][0]].dot(feature_conv[batch_size, :, :, :].reshape((c, h * w)))
        cam = cam.reshape(h, w)

        cam_img = (cam - cam.min()) / (cam.max() - cam.min())

        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam


# def calculate_CAM(args, imgs, out, conv_features, weight_softmax, num, phase, epoch):
#     # calculate CAM
#     # print(out.shape)
#     # print(params[-2].data.cpu().numpy().shape)
#     # print(feature_blobs_test[0].shape)
#
#     h_x = F.softmax(out, dim=1).data.squeeze()
#     # print(h_x.shape)
#     _, idx = torch.sort(h_x, descending=True)
#     idx = idx.cpu().numpy()
#     conv_features = conv_features.detach().cpu().numpy()
#     CAMs = returnCAM(conv_features, weight_softmax, [idx])  # list([224, 224, 3]...)
#     bz, c, height, width = imgs.shape  # [bz, channel 3, height 224, width 224]
#     for batch_size in range(bz):
#         num += 1
#         heatmap = cv2.applyColorMap(cv2.resize(CAMs[batch_size], (width, height)), cv2.COLORMAP_JET)  # [224, 224, 3]
#         # heatmap[CAMs[batch_size] <= 100] = 0
#         imgs_numpy = imgs[batch_size, :, :, :]
#         imgs_numpy = imgs_numpy.permute(1, 2, 0).contiguous()
#         imgs_numpy = imgs_numpy[:, :, [2, 1, 0]]  # bgr -> rgb
#         imgs_numpy = imgs_numpy.detach().cpu().numpy()
#         imgs_numpy = imgs_numpy - np.min(imgs_numpy)
#         imgs_numpy = imgs_numpy / (np.max(imgs_numpy))
#         imgs_numpy = imgs_numpy * 255
#         result = 0.3 * heatmap + imgs_numpy
#
#         if phase == 'train':
#             path = '/home/zhongtao/code/Self-Cure-Network/AttSA/train/{}_{:>05d}_{}.png'.format(args.sourceDataset, num,
#                                                                                                 epoch)
#             cv2.imwrite(path, result)  # 保存结果
#         else:
#             path = '/home/zhongtao/code/Self-Cure-Network/AttSA/test/{}_{:>05d}_{}.png'.format(args.targetDataset, num,
#                                                                                                epoch)
#             cv2.imwrite(path, result)  # 保存结果


def make_histogram(args, img, phase, num, turn):
    b = img.shape[0]
    img = img.cpu().detach().numpy()
    if phase == 'train':
        for i in range(b):
            plt.figure()
            num += 1
            save_path = '/home/zhongtao/code/Self-Cure-Network/attention/train'
            arr = img[i, :, :, :]
            arr = arr.flatten()
            avg = np.average(arr)
            plt.hist(arr, bins=100, range=[0, 1], facecolor='green', alpha=0.75)
            plt.title(str(avg))
            plt.savefig('{}/{}_{}_{:>05d}.png'.format(save_path, args.sourceDataset, turn, num))
            plt.close()
    else:
        for i in range(b):
            plt.figure()
            num += 1
            save_path = '/home/zhongtao/code/Self-Cure-Network/attention/test'
            arr = img[i, :, :, :]
            arr = arr.flatten()
            avg = np.average(arr)
            plt.hist(arr, bins=100, range=[0, 1], facecolor='blue', alpha=0.75)
            plt.title(str(avg))
            plt.savefig('{}/{}_{}_{:>05d}.png'.format(save_path, args.targetDataset, turn, num))
            plt.close()


def show_feature_map(args, image, label, conv_features, num=0, epoch=0, stage=0):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    for i in range(conv_features.shape[0]):
        num += 1
        img = image[i, :, :, :]
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # minmax归一化处理
        img = np.uint8(255 * img)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
        # img = img.squeeze(0)
        heat = conv_features[i, :, :, :]
        # heat = conv_features.squeeze(0)  # 降维操作,尺寸变为(2048,7,7)
        heat_mean = torch.sum(heat, dim=0)  # 对各卷积层(2048)求平均值,尺寸变为(7,7)

        # print(heat_mean)
        # print(torch.sum(heat_mean))
        heatmap = heat_mean.detach().cpu().numpy()  # 转换为numpy数组
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # minmax归一化处理
        heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))  # 变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
        heatmap = np.uint8(255 * heatmap)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 颜色变换
        # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
        superimg = heatmap * 0.4 + img  # 图像叠加，注意翻转通道，cv用的是bgr
        path = '/home/zhongtao/code/CrossDomainFER/my_method/CAM/test/{}_{}_{:>02d}_{:>05d}_{:>01d}_{}.png'.format(
            args.sourceDataset, args.targetDataset, epoch, num, stage, label[i].item())
        cv2.imwrite(path, superimg)  # 保存结果


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class AngularSoftmaxWithLoss(nn.Module):
    """"""

    def __init__(self, gamma=0):
        super(AngularSoftmaxWithLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.iter += 1
        target = target.view(-1, 1)

        index = input[0].data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = Variable(index.byte())

        # Tricks
        # output(θyi) = (lambda * cos(θyi) + (-1) ** k * cos(m * θyi) - 2 * k)) / (1 + lambda)
        #             = cos(θyi) - cos(θyi) / (1 + lambda) + Phi(θyi) / (1 + lambda)
        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = input[0] * 1.0
        output[index] -= input[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input[1][index] * 1.0 / (1 + self.lamb)

        # softmax loss
        logit = F.log_softmax(output)
        logit = logit.gather(1, target).view(-1)
        pt = logit.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features=7, out_features=7, s=30.0, m=0.5, t=5, k=1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.t = t
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False).cuda()

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''

        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        '''

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x_temp = F.normalize(x, dim=1)
        x = self.fc(x)
        x_pow = x * x
        x_pow_sum = torch.sum(x_pow, dim=1)
        x_mod = torch.sqrt(x_pow_sum)

        b = torch.ones(len(labels)) * self.t
        b = b.cuda()
        x_mod = torch.where(x_mod < self.t, x_mod, b)

        wf = self.fc(x_temp)
        cosin = torch.diagonal(wf.transpose(0, 1)[labels])
        theta = torch.acos(cosin)
        f = -self.k * (theta - math.pi / 2)
        up = x_mod * f

        cosin_unrelate = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)],
                                   dim=0)
        f_none = -self.k * (torch.acos(cosin_unrelate) - math.pi / 2) + self.m
        excl = x_mod.unsqueeze(dim=1) * f_none
        denominator = torch.exp(up) + torch.sum(torch.exp(excl), dim=1)
        L = up - torch.log(denominator)
        return -torch.mean(L)


def VisualizationForTwoDomain(path, model, source_dataloader, target_dataloader):
    '''Feature Visualization in Source and Target Domain.'''

    model.eval()

    Feature_Source, Label_Source = [], []

    # Get Feature and Label in Source Domain
    for batch_i, (imgs, label, indexes) in enumerate(source_dataloader):
        imgs, label = imgs.cuda(), label.cuda()
        with torch.no_grad():
            output, _, _, _, _, _, feature = model(imgs)

        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    Feature_Target, Label_Target = [], []
    iter_target_dataloader = iter(target_dataloader)

    # Get Feature and Label in Target Domain
    for batch_index, (input, label, _) in enumerate(iter_target_dataloader):
        input, label = input.cuda(), label.cuda()
        with torch.no_grad():
            output, _, _, _, _, _, feature = model(input)

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    Label_Target += 7

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target))

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    plot_num = 13000
    embedding = tsne.fit_transform(Feature[:plot_num, :])

    # Draw Visualization of Feature
    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray', \
              7: 'red', 8: 'blue', 9: 'olive', 10: 'green', 11: 'orange', 12: 'purple', 13: 'darkslategray'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral', \
              7: 'Surprised', 8: 'Fear', 9: 'Disgust', 10: 'Happy', 11: 'Sad', 12: 'Angry', 13: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    # ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label[:plot_num] == i][:, 0], data_norm[Label[:plot_num] == i][:, 1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20,
                                     label=labels[i], marker="o", alpha=0.4, linewidth=0.5)

        data_target_x, data_target_y = data_norm[Label[:plot_num] == (i + 7)][:, 0], data_norm[
                                                                                         Label[:plot_num] == (i + 7)][:,
                                                                                     1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30,
                                     label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            source_legend = source_scatter
            target_legend = target_scatter

    # tmp = [0, 1]
    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in tmp ], loc='upper right', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(7)], loc='upper right', prop = {'size':8})
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
    #                 loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop={'size': 7},
    #            bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
    # plt.gca().add_artist(l1)

    plt.savefig(fname=path, format="png", bbox_inches='tight')


def VisualizationForOneDomain(path, model, dataloader):
    '''Feature Visualization in Source and Target Domain.'''

    model.eval()

    Feature_Target, Label_Target = [], []
    iter_target_dataloader = iter(dataloader)

    # Get Feature and Label in Target Domain
    for batch_index, (input, label, _) in enumerate(iter_target_dataloader):
        input, label = input.cuda(), label.cuda()
        with torch.no_grad():
            output, _, _, _, _, _, feature = model(input)

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    Feature = Feature_Target
    Label = Label_Target

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3, )
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    # ax = plt.subplot(111)

    for i in range(7):

        data_target_x, data_target_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30,
                                     label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            target_legend = target_scatter

    # tmp = [0, 1]
    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in tmp ], loc='upper right', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(7)], loc='upper right', prop = {'size':8})
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop = {'size':8})

    # l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
    #                 loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    # plt.legend([source, target], ['Source Domain', 'Target Domain'], loc='upper left', prop={'size': 7},
    #            bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
    # plt.gca().add_artist(l1)

    plt.savefig(fname=path, format="png", bbox_inches='tight')


def VisualizationForDistrubution(x, feature1=None, feature2=None, feature3=None, feature4=None):
    tsne = TSNE(n_components=2)
    f = tsne.fit_transform(x.cpu().numpy())
    f1 = tsne.fit_transform(feature1.cpu().numpy())
    f2 = tsne.fit_transform(feature2.cpu().numpy())
    f3 = tsne.fit_transform(feature3.cpu().numpy())
    f4 = tsne.fit_transform(feature4.cpu().numpy())

    data_min, data_max = np.min(f, 0), np.max(f, 0)
    f = (f - data_min) / (data_max - data_min)
    data_min, data_max = np.min(f1, 0), np.max(f1, 0)
    f1 = (f1 - data_min) / (data_max - data_min)
    data_min, data_max = np.min(f2, 0), np.max(f2, 0)
    f2 = (f2 - data_min) / (data_max - data_min)
    data_min, data_max = np.min(f3, 0), np.max(f3, 0)
    f3 = (f3 - data_min) / (data_max - data_min)
    data_min, data_max = np.min(f4, 0), np.max(f4, 0)
    f4 = (f4 - data_min) / (data_max - data_min)

    plt.figure()
    # ax = sns.kdeplot(x=f[:, 0], y=f[:, 1], cmap="Reds", fill=True)
    ax = sns.kdeplot(f, fill=True, color="r", bw_adjust=2, legend='origin')
    ax.legend(['origin'])
    plt.savefig(fname='./123/x.png', format="png", bbox_inches='tight')
    plt.figure()
    # ax = sns.kdeplot(x=f1[:, 0], y=f1[:, 1], cmap="Blues", fill=True)
    ax = sns.kdeplot(f1, fill=True, color="b", bw_adjust=2, legend='1')
    ax.legend(['1'])
    plt.savefig(fname='./123/1.png', format="png", bbox_inches='tight')
    plt.figure()
    # ax = sns.kdeplot(x=f2[:, 0], y=f2[:, 1], cmap="Greens", fill=True)
    ax = sns.kdeplot(f2, fill=True, color="g", bw_adjust=2, legend='2')
    ax.legend(['2'])
    plt.savefig(fname='./123/2.png', format="png", bbox_inches='tight')
    plt.figure()
    # ax = sns.kdeplot(x=f3[:, 0], y=f3[:, 1], cmap="Oranges", fill=True)
    ax = sns.kdeplot(f3, fill=True, color="orange", bw_adjust=2, legend='3')
    ax.legend(['3'])
    plt.savefig(fname='./123/3.png', format="png", bbox_inches='tight')
    plt.figure()
    # ax = sns.kdeplot(x=f4[:, 0], y=f4[:, 1], cmap="Greys", fill=True)
    ax = sns.kdeplot(f4, fill=True, color="grey", bw_adjust=2, legend='4')
    ax.legend(['4'])
    plt.savefig(fname='./123/4.png', format="png", bbox_inches='tight')
    plt.figure()

    # kwargs = dict(histtype='stepfilled', alpha=0.3, bins=20)
    # plt.hist(f1, **kwargs)
    # plt.hist(f2, **kwargs)
    # plt.hist(f3, **kwargs)
    # plt.hist(f4, **kwargs)

    # ax = sns.kdeplot(x=f[:, 0], y=f[:, 1], cmap="Reds", fill=True)
    ax = sns.kdeplot(f, fill=True, color="r", bw_adjust=2, legend='origin')
    # ax = sns.kdeplot(x=f1[:, 0], y=f1[:, 1], cmap="Blues", fill=True)
    ax = sns.kdeplot(f1, fill=True, color="b", bw_adjust=2, legend='1')
    # ax = sns.kdeplot(x=f2[:, 0], y=f2[:, 1], cmap="Greens", fill=True)
    ax = sns.kdeplot(f2, fill=True, color="g", bw_adjust=2, legend='2')
    # ax = sns.kdeplot(x=f3[:, 0], y=f3[:, 1], cmap="Oranges", fill=True)
    ax = sns.kdeplot(f3, fill=True, color="orange", bw_adjust=2, legend='3')
    # ax = sns.kdeplot(x=f4[:, 0], y=f4[:, 1], cmap="Greys", fill=True)
    ax = sns.kdeplot(f4, fill=True, color="grey", bw_adjust=2, legend='4')
    ax.legend(['origin','1', '2', '3', '4'])
    plt.savefig(fname='./123/test.png', format="png", bbox_inches='tight')

    # plt.figure()
    # plt.hist2d(f1[:, 0], f1[:, 1], bins=30, cmap='Blues')
    # plt.savefig(fname='test1.png', format="png", bbox_inches='tight')
    # plt.figure()
    # plt.hist2d(f2[:, 0], f2[:, 1], bins=30, cmap='Blues')
    # plt.savefig(fname='test2.png', format="png", bbox_inches='tight')
    # plt.figure()
    # plt.hist2d(f3[:, 0], f3[:, 1], bins=30, cmap='Blues')
    # plt.savefig(fname='test3.png', format="png", bbox_inches='tight')
    # plt.figure()
    # plt.hist2d(f4[:, 0], f4[:, 1], bins=30, cmap='Blues')
    # plt.savefig(fname='test4.png', format="png", bbox_inches='tight')

def AttentionHeatMap(structure_encoding, phase):
    metric = CosineSimilarity(zero_diagonal=False)
    # metric = CosineSimilarity()

    # structure_vector 就是你的transformer的token
    for i in range(structure_encoding.shape[0]):
        structure_vector = structure_encoding[i].view(196, 512)  # 1 98 256 -> 98 256

        metric.clear()
        metric.update(structure_vector.detach().cpu().numpy())
        square_matrix = metric.eval()
        plt.figure()
        sns.heatmap(square_matrix, square=True, vmin=0, vmax=1)
        plt.xlabel('number of token')
        plt.ylabel('number of token')
        path = '/home/zhongtao/code/CrossDomainFER/my_method/heatmap__{}.png'.format(phase)
        # fig = ax.get_figure()
        plt.savefig(fname=path, format="png", bbox_inches='tight')
