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
from tqdm import trange
from tqdm.contrib import tenumerate

from Datasets import RafDataSet, SFEWDataSet, JAFFEDataSet, FER2013DataSet, ExpWDataSet, AffectNetDataSet, \
    FER2013PlusDataSet
from Utils import *

parser = argparse.ArgumentParser(description='Expression Classification Training')

parser.add_argument('--Log_Name', type=str, default='train', help='Logs Name')
parser.add_argument('--OutputPath', default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints', type=str,
                    help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Network', type=str, default='AConv+DHSA+global(x)',
                    choices=['Baseline', 'AConv', 'AConv+DHSA', 'AConv+DHSA+global', 'AConv+DHSA+global(x)', 'unified',
                             'test'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model',
                    default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB+FER2013Plus+SFEW_AffectNet_AConv+DHSA+global(x).pkl')
parser.add_argument('--pretrained', type=str,
                    default='/home/zhongtao/code/Self-Cure-Network/models/vit_base_patch16_224.pth',
                    help='Pretrained weights')
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=224, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAFDB+FER2013Plus+SFEW',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'FER2013Plus', 'AffectNet', 'ExpW', 'RAFDB+SFEW',
                             'RAFDB+FER2013Plus', 'FER2013Plus+SFEW', 'RAFDB+FER2013Plus+SFEW'])
parser.add_argument('--targetDataset', type=str, default='AffectNet',
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
parser.add_argument('--batch_size1', type=int, default=16, help='input batch size for training (default: 64)')
parser.add_argument('--batch_size2', type=int, default=32, help='input batch size for training (default: 64)')
parser.add_argument('--batch_size3', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=True, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 60)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='SGD weight decay (default: 0.0001)')

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--isSave', type=str2bool, default=True, help='whether to save model')
parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')

parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
parser.add_argument('--seed', type=int, default=2022, help='random seed (default: 1)')

num = 0


def Train(args, model, ad_net, train_dataloader1, train_dataloader2, train_dataloader3, optimizer, optimizer_ad,
          scheduler, scheduler_ad,
          epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i
                                                                                                 in
                                                                                                 range(7)]

    loss, dan_loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    # mmd_loss = AverageMeter()
    # center_loss = AverageMeter()
    # loss_norm = AverageMeter()

    end = time.time()
    num_ADNet = 0
    ad_net.train()

    iter_source_dataloader1 = iter(train_dataloader1)
    iter_source_dataloader2 = iter(train_dataloader2)
    iter_source_dataloader3 = iter(train_dataloader3)

    num_iter = len(train_dataloader1) if (len(train_dataloader1) > len(train_dataloader2)) else len(train_dataloader2)

    for batch_i in trange(num_iter):
        try:
            data_source1, label_source1, _ = next(iter_source_dataloader1)
        except:
            iter_source_dataloader1 = iter(train_dataloader1)
            data_source1, label_source1, _ = next(iter_source_dataloader1)

        try:
            data_source2, label_source2, _ = next(iter_source_dataloader2)
        except:
            iter_source_dataloader2 = iter(train_dataloader2)
            data_source2, label_source2, _ = next(iter_source_dataloader2)

        try:
            data_source3, label_source3, _ = next(iter_source_dataloader3)
        except:
            iter_source_dataloader3 = iter(train_dataloader3)
            data_source3, label_source3, _ = next(iter_source_dataloader3)

        data_time.update(time.time() - end)

        data_source1 = data_source1.cuda()
        data_source2 = data_source2.cuda()
        data_source3 = data_source3.cuda()
        label_source1 = label_source1.cuda()
        label_source2 = label_source2.cuda()
        label_source3 = label_source3.cuda()
        data_source = torch.cat((data_source1, data_source2, data_source3), 0)
        # data_source = torch.cat((data_source1, data_source2), 0)
        label_source = torch.cat((label_source1, label_source2, label_source3), 0)
        # label_source = torch.cat((label_source1, label_source2), 0)

        # Forward propagation

        output, featureMap, featureMap1, featureMap2, featureMap3, featureMap4, feature = model(data_source)

        batch_time.update(time.time() - end)

        # Compute Loss
        global_cls_loss_ = LabelSmoothLoss()(output, label_source)
        dan_loss_ = DANN(feature, ad_net)
        # center_loss_ = CenterLoss(num_classes=7, feat_dim=1024)(feature, label)
        # norm_loss_ = torch.mean(feature.norm(p=2, dim=1))

        # loss_ = global_cls_loss_ + 0.01 * norm_loss_
        loss_ = global_cls_loss_ + dan_loss_

        # Log Adversarial Network Accuracy

        adnet_output = ad_net(feature)

        adnet_output = adnet_output.cpu().data.numpy()
        adnet_output[adnet_output > 0.5] = 1
        adnet_output[adnet_output <= 0.5] = 0
        num_ADNet += np.sum(adnet_output[:args.batch_size1]) + (
                args.batch_size2 - np.sum(adnet_output[args.batch_size1:]))

        # Back Propagation
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        loss_.backward()
        optimizer.step()
        optimizer_ad.step()

        # Compute accuracy, recall and loss
        Compute_Accuracy(args, output, label_source, acc, prec, recall)

        # Logs loss
        loss.update(float(global_cls_loss_.cpu().data.item()))
        dan_loss.update(float(dan_loss_.cpu().data.item()))
        # mmd_loss.update(float(mmd_loss_.cpu().data.item()))
        # center_loss.update(float(center_loss_.cpu().data.item()))
        # loss_norm.update(float(norm_loss_.cpu().data.item()))

        end = time.time()

    scheduler.step()
    scheduler_ad.step()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    writer.add_scalar('Loss', loss.avg, epoch)

    LoggerInfo = '''
    [Tain]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, scheduler.get_lr(), data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    AdversarialNet Acc {0:.4f} Acc_avg {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f} Norm Loss {5:.4f} Total Loss {loss:.4f}\n'''.format(
        num_ADNet / (2.0 * args.batch_size * batch_i), acc_avg, prec_avg, recall_avg, f1_avg, 0, loss=loss.avg)

    # LoggerInfo += '''    MMD Loss {0:.4f} Total Loss {1:.4f}'''.format(mmd_loss.avg, loss.avg)

    print(LoggerInfo)


def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc_Source, Best_Acc_Target, epoch, writer):
    """Test."""
    global num

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    iter_target_dataloader = iter(test_target_dataloader)

    # # Test on Source Domain
    # acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
    #                                                                                              range(7)]
    # loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
    #
    # end = time.time()
    # for batch_index, (input, label, _) in enumerate(iter_source_dataloader):
    #     data_time.update(time.time() - end)
    #
    #     input, label = input.cuda(), label.cuda()
    #
    #     with torch.no_grad():
    #         end = time.time()
    #         output, featureMap, featureMap1, featureMap2, featureMap3, featureMap4, feature = model(input)
    #         # feature, output, _, _, _, _ = model(input)
    #         batch_time.update(time.time() - end)
    #
    #     loss_ = LabelSmoothLoss()(output, label)
    #
    #     # Compute accuracy, precision and recall
    #     Compute_Accuracy(args, output, label, acc, prec, recall)
    #
    #     # Logs loss
    #     loss.update(float(loss_.cpu().data.numpy()))
    #
    #     end = time.time()
    #
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #
    # writer.add_scalar('Test_Recall_SourceDomain', recall_avg, epoch)
    # writer.add_scalar('Test_Accuracy_SourceDomain', acc_avg, epoch)
    #
    # LoggerInfo = '''
    # [Test (Source Domain)]:
    # Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    # Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)
    #
    # LoggerInfo += AccuracyInfo
    #
    # LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    # Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)
    #
    # print(LoggerInfo)
    #
    # if acc_avg > Best_Acc_Source:
    #     Best_Acc_Source = acc_avg
    # print('[Save] Best Accuracy: %.4f.' % Best_Acc_Source)

    # Save Checkpoints
    # if recall_avg > Best_Recall:
    #     Best_Recall = recall_avg
    #     print('[Save] Best Recall: %.4f.' % Best_Recall)
    #
    #     if isinstance(model, nn.DataParallel):
    #         torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
    #     else:
    #         torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, label, _) in tenumerate(iter_target_dataloader):
        data_time.update(time.time() - end)

        input, label = input.cuda(), label.cuda()

        with torch.no_grad():
            end = time.time()
            output, featureMap, featureMap1, featureMap2, featureMap3, featureMap4, feature = model(input)
            # feature, output, _, _, _, _ = model(input)
            batch_time.update(time.time() - end)

        loss_ = LabelSmoothLoss()(output, label)
        # show_feature_map(args, input, featureMap, num, epoch, stage=0)
        # show_feature_map(args, input, featureMap1, num, epoch, stage=1)
        # show_feature_map(args, input, featureMap2, num, epoch, stage=2)
        # show_feature_map(args, input, featureMap3, num, epoch, stage=3)
        # show_feature_map(args, input, featureMap4, num, epoch, stage=4)
        # num += input.shape[0]

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Logs loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_TargetDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_TargetDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Target Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)
    num = 0

    if acc_avg > Best_Acc_Target and not args.isTest:
        Best_Acc_Target = acc_avg
        print('[Save] Best Accuracy: %.4f.' % Best_Acc_Target)

    return Best_Acc_Source, Best_Acc_Target


def main():
    """Main."""

    # Parse Argument
    args = parser.parse_args()
    # if not args.isTest:
    #     sys.stdout = Logger(
    #         osp.join('./Logs/', '{}_{}_{}.txt'.format(args.sourceDataset, args.targetDataset, args.Network)))
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
    print('Batch Size1: %d' % args.batch_size1)
    print('Batch Size2: %d' % args.batch_size2)

    print('================================================')

    if args.isTest:
        print('Test Model.')
    else:
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)
        print('Number of classes : %d' % args.class_num)

    print('================================================')

    # Bulid Dataloder
    print("Buliding Train and Test Dataloader...")
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.sourceDataset == 'FER2013Plus+SFEW':
        train_dataset1 = FER2013PlusDataSet(args.fer2013plus_path, phase='train', transform=data_transforms,
                                            basic_aug=True)
        train_dataset2 = SFEWDataSet(args.sfew_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'RAFDB+SFEW':
        train_dataset1 = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)
        train_dataset2 = SFEWDataSet(args.sfew_path, phase='train', transform=data_transforms, basic_aug=True)
    elif args.sourceDataset == 'RAFDB+FER2013Plus':
        train_dataset1 = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)
        train_dataset2 = FER2013PlusDataSet(args.fer2013plus_path, phase='train', transform=data_transforms,
                                            basic_aug=True)
    elif args.sourceDataset == 'RAFDB+FER2013Plus+SFEW':
        train_dataset1 = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)
        train_dataset2 = FER2013PlusDataSet(args.fer2013plus_path, phase='train', transform=data_transforms,
                                            basic_aug=True)
        train_dataset3 = SFEWDataSet(args.sfew_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset1.__len__())
    print('The Source Train dataset distribute:', train_dataset1.__distribute__())
    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                                batch_size=args.batch_size1,
                                                num_workers=args.workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=True)
    print('Train set size:', train_dataset2.__len__())
    print('The Source Train dataset distribute:', train_dataset2.__distribute__())
    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                                batch_size=args.batch_size2,
                                                num_workers=args.workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=True)
    print('Train set size:', train_dataset3.__len__())
    print('The Source Train dataset distribute:', train_dataset3.__distribute__())
    train_loader3 = torch.utils.data.DataLoader(train_dataset3,
                                                batch_size=args.batch_size3,
                                                num_workers=args.workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=True)

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
    else:
        val_dataset_source = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)

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

    # print('Validation Source set size:', val_dataset_source.__len__())
    # print('The Validation Source dataset distribute:', val_dataset_source.__distribute__())
    print('Validation Target set size:', val_dataset_target.__len__())
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
    print('Done!')

    print('================================================')

    # Bulid Adversarial Network
    print('Building Adversarial Network...')
    ad_net = BulidAdversarialNetwork(args, 1024, args.class_num)
    print('Done!')

    print('================================================')

    # Set Optimizer
    print('Buliding Optimizer...')
    params = model.parameters()
    optimizer = optim.AdamW(params, betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    params_ad = ad_net.parameters()
    optimizer_ad = optim.AdamW(params_ad, betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_ad = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Acc_Source, Best_Acc_Target = 0, 0

    # Running Experiment
    print("Run Experiment...")

    writer = SummaryWriter(os.path.join('/home/zhongtao/code/CrossDomainFER/my_method/LogInfo',
                                        '{}_{}_{}'.format(args.sourceDataset, args.targetDataset, args.Network)))

    for epoch in range(1, args.epochs + 1):
        # if not args.isTest:
        #     Train(args, model, ad_net, train_loader1, train_loader2, train_loader3, optimizer, optimizer_ad, scheduler, scheduler_ad,
        #           epoch, writer)

        Best_Acc_Source, Best_Acc_Target = Test(args, model, val_loader_source, val_loader_target, Best_Acc_Source,
                                                Best_Acc_Target, epoch, writer)

        # if args.showFeature:
        #     VisualizationForTwoDomain(
        #         '/home/zhongtao/code/CrossDomainFER/my_method/visualization/{}_{}_{:>02d}.pdf'.format(
        #             args.sourceDataset, args.targetDataset, epoch), model, train_loader, val_loader_target)

        # torch.cuda.empty_cache()

    writer.close()
    print('Best Accuarcy on Source Domain:%.4f' % (Best_Acc_Source))
    print('Best Accuarcy on Target Domain:%.4f' % (Best_Acc_Target))


if __name__ == '__main__':
    main()
