import os
import random
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Datasets import RafDataSet, SFEWDataSet, JAFFEDataSet, FER2013DataSet, ExpWDataSet, AffectNetDataSet, \
    FER2013PlusDataSet
from Utils import *

parser = argparse.ArgumentParser(description='Expression Classification Training')

parser.add_argument('--Log_Name', type=str, default='test', help='Logs Name')
parser.add_argument('--OutputPath', default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints', type=str,
                    help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Network', type=str, default='AConv+DHSA+global(x)',
                    choices=['Baseline', 'AConv', 'AConv+DHSA', 'AConv+DHSA+global', 'AConv+DHSA+global(x)', 'unified'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model',
                    default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB_JAFFE_AConv+DHSA+global(x)_test.pkl')
parser.add_argument('--GPU_ID', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=224, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAFDB',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'FER2013Plus', 'AffectNet', 'ExpW'])
parser.add_argument('--targetDataset', type=str, default='JAFFE',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'FER2013Plus', 'AffectNet', 'JAFFE', 'ExpW'])
parser.add_argument('--raf_path', type=str, default='/home/zhongtao/datasets/RAFDB',
                    help='Raf-DB dataset path.')
parser.add_argument('--jaffe-path', type=str, default='/home/zhongtao/datasets/jaffedbase',
                    help='JAFFE dataset path.')
parser.add_argument('--fer2013-path', type=str, default='/home/zhongtao/datasets/FER2013',
                    help='FER2013 dataset path.')
parser.add_argument('--fer2013plus-path', type=str, default='/home/zhongtao/datasets/FER2013+',
                    help='FER2013Plus dataset path.')
parser.add_argument('--expw-path', type=str, default='/home/zhongtao/datasets/ExpW',
                    help='ExpW dataset path.')
parser.add_argument('--sfew-path', type=str, default='/home/zhongtao/datasets/SFEW2.0',
                    help='SFEW dataset path.')
parser.add_argument('--affectnet-path', type=str, default='/home/zhongtao/datasets/AffectNet',
                    help='AffectNet dataset path.')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 60)')

parser.add_argument('--showFeature', type=str2bool, default=True, help='whether to show feature')

parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 1)')

num = 0


# def Train(args, model, train_dataloader, optimizer, scheduler, len_train):
#     """Train."""
#
#     model.train()
#     torch.autograd.set_detect_anomaly(True)
#
#     acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
#                                                                                                  range(7)]
#     loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
#
#     end = time.time()
#     grad = 0
#     for batch_i, (imgs, label, indexes) in enumerate(train_dataloader):
#         imgs, label = imgs.cuda(), label.cuda()
#         data_time.update(time.time() - end)
#
#         # Forward propagation
#         end = time.time()
#         output, featureMap, featureMap1, featureMap2, featureMap3, featureMap4, feature = model(imgs)
#         # feature, output, local1, local2, local3, local4 = model(imgs)
#         batch_time.update(time.time() - end)
#
#         # Compute Loss
#         # global_cls_loss_ = torch.nn.CrossEntropyLoss()(output, label)
#         global_cls_loss_ = LabelSmoothLoss()(output, label)
#         # local_cls_loss_ = LabelSmoothLoss()(local1, label) + LabelSmoothLoss()(local2, label) + LabelSmoothLoss()(
#         #     local3, label) + LabelSmoothLoss()(local4, label)
#         # score_loss_ = torch.mean(score)
#
#         # loss_ = global_cls_loss_ + mask_loss_
#         # loss_ = global_cls_loss_ + local_cls_loss_
#         loss_ = global_cls_loss_
#
#         # Back Propagation
#         optimizer.zero_grad()
#         loss_.backward()
#         optimizer.step()
#         grad = grad + model.fc.weight.grad.sum()
#
#         # Compute accuracy, recall and loss
#         Compute_Accuracy(args, output, label, acc, prec, recall)
#
#         # Logs loss
#         loss.update(float(loss_.cpu().data.item()))
#
#         end = time.time()
#
#     scheduler.step()
#
#     AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
#
#     LoggerInfo = '''
#     [Tain]:
#     Epoch {0}
#     Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
#     Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
#     Learning Rate {1}\n'''.format(0, scheduler.get_lr(), data_time=data_time, batch_time=batch_time)
#
#     LoggerInfo += AccuracyInfo
#
#     LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f} Total Loss {loss:.4f}'''.format(
#         acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)
#
#     print(LoggerInfo)
#     print(grad / len_train)


def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc_Source, Best_Acc_Target, len_source,
         len_target):
    """Test."""
    global num

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    iter_target_dataloader = iter(test_target_dataloader)
    conf_matrix = torch.zeros(7, 7)

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, label, _) in enumerate(iter_target_dataloader):
        data_time.update(time.time() - end)

        input, label = input.cuda(), label.cuda()

        with torch.no_grad():
            end = time.time()
            output, featureMap, featureMap1, featureMap2, featureMap3, featureMap4, feature = model(input)
            batch_time.update(time.time() - end)

        loss_ = LabelSmoothLoss()(output, label)
        # for i in range(512):
        #     show_feature_map(args, input, feature[:,i:i+1,:,:], num, 0, stage=i)
        # show_feature_map(args, input, label, featureMap, num, 0, stage=0)
        # show_feature_map(args, input, featureMap1, num, 0, stage=1)
        # show_feature_map(args, input, featureMap2, num, 0, stage=2)
        # show_feature_map(args, input, featureMap3, num, 0, stage=3)
        # show_feature_map(args, input, featureMap4, num, 0, stage=4)
        num += input.shape[0]

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)
        conf_matrix = confusion_matrix(output, label, conf_matrix)
        # get_wrong_image(args, output, label)

        # Logs loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    # print_confusion_matrix(conf_matrix)

    LoggerInfo = '''
    [Test (Target Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)
    num = 0

    if acc_avg > Best_Acc_Target:
        Best_Acc_Target = acc_avg

    return Best_Acc_Source, Best_Acc_Target


def main():
    """Main."""

    # Parse Argument
    args = parser.parse_args()
    # sys.stdout = Logger(
    #     osp.join('./Logs/', '{}_{}_{}_test.txt'.format(args.sourceDataset, args.targetDataset, args.Network)))
    if args.seed:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('set seed:{}'.format(args.seed))

    # Experiment Information
    print('Logs Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Network: %s' % args.Network)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Batch Size: %d' % args.batch_size)

    print('================================================')

    print('Test Model.')

    print('================================================')

    # Bulid Dataloder
    print("Buliding Train and Test Dataloader...")
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.sourceDataset == 'RAFDB':
        train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'FER2013':
        train_dataset = FER2013DataSet(args.fer2013_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'FER2013Plus':
        train_dataset = FER2013PlusDataSet(args.fer2013plus_path, phase='train', transform=data_transforms,
                                           basic_aug=True)
    elif args.sourceDataset == 'SFEW':
        train_dataset = SFEWDataSet(args.sfew_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'ExpW':
        train_dataset = ExpWDataSet(args.expw_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'AffectNet':
        train_dataset = AffectNetDataSet(args.affectnet_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    len_train = train_dataset.__len__()
    print('The Source Train dataset distribute:', train_dataset.__distribute__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])])

    if args.sourceDataset == 'RAFDB':
        val_dataset_source = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'FER2013':
        val_dataset_source = FER2013DataSet(args.fer2013_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'FER2013Plus':
        val_dataset_source = FER2013PlusDataSet(args.fer2013plus_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'SFEW':
        val_dataset_source = SFEWDataSet(args.sfew_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'ExpW':
        val_dataset_source = ExpWDataSet(args.expw_path, phase='test', transform=data_transforms_val)
    elif args.sourceDataset == 'AffectNet':
        val_dataset_source = AffectNetDataSet(args.affectnet_path, phase='test', transform=data_transforms_val)

    if args.targetDataset == 'RAFDB':
        val_dataset_target = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'JAFFE':
        val_dataset_target = JAFFEDataSet(args.jaffe_path, transform=data_transforms_val)
    elif args.targetDataset == 'FER2013':
        val_dataset_target = FER2013DataSet(args.fer2013_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'FER2013Plus':
        val_dataset_target = FER2013PlusDataSet(args.fer2013plus_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'SFEW':
        val_dataset_target = SFEWDataSet(args.sfew_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'ExpW':
        val_dataset_target = ExpWDataSet(args.expw_path, phase='test', transform=data_transforms_val)
    elif args.targetDataset == 'AffectNet':
        val_dataset_target = AffectNetDataSet(args.affectnet_path, phase='test', transform=data_transforms_val)

    print('Validation Source set size:', val_dataset_source.__len__())
    len_source = val_dataset_source.__len__()
    print('The Validation Source dataset distribute:', val_dataset_source.__distribute__())
    print('Validation Target set size:', val_dataset_target.__len__())
    len_target = val_dataset_target.__len__()
    print('The Validation Target dataset distribute:', val_dataset_target.__distribute__())

    val_loader_source = torch.utils.data.DataLoader(val_dataset_source,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)
    val_loader_target = torch.utils.data.DataLoader(val_dataset_target,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers,
                                                    shuffle=False,
                                                    pin_memory=True)
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Buliding Model...')
    model = BulidModel(args)
    params = model.parameters()
    optimizer = optim.AdamW(params, betas=(0.9, 0.999), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Acc_Source, Best_Acc_Target = 0, 0

    # Running Experiment
    print("Run Experiment...")

    # Train(args, model, train_loader, optimizer, scheduler, len_train)

    # model = BulidModel(args)

    Best_Acc_Source, Best_Acc_Target = Test(args, model, val_loader_source, val_loader_target, Best_Acc_Source,
                                            Best_Acc_Target, len_source, len_target)

    # if args.showFeature:
    #     VisualizationForTwoDomain(
    #         '/home/zhongtao/code/CrossDomainFER/my_method/visualization/{}_{}_{:>02d}.png'.format(
    #             args.sourceDataset, args.targetDataset, 0), model, train_loader, val_loader_target)

        # VisualizationForOneDomain(
        #     '/home/zhongtao/code/CrossDomainFER/my_method/visualization/{}_{:>02d}.png'.format(
        #         args.targetDataset, 0), model, val_loader_target)

    print('Best Accuarcy on Source Domain:%.4f' % (Best_Acc_Source))
    print('Best Accuarcy on Target Domain:%.4f' % (Best_Acc_Target))


if __name__ == '__main__':
    main()
