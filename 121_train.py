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

parser.add_argument('--Log_Name', type=str, default='train', help='Logs Name')
parser.add_argument('--OutputPath', default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints', type=str,
                    help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Network', type=str, default='Baseline',
                    choices=['Baseline', 'AConv', 'AConv+DHSA', 'AConv+DHSA+global', 'AConv+DHSA+global(x)', 'unified'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=224, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='SFEW',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'FER2013Plus', 'AffectNet', 'ExpW'])
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
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

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


def Train(args, model, train_dataloader, optimizer, scheduler, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_i, (imgs, label, indexes) in enumerate(train_dataloader):
        imgs, label = imgs.cuda(), label.cuda()
        data_time.update(time.time() - end)

        # Forward propagation
        end = time.time()
        output, featureMap, featureMap1, featureMap2, featureMap3, featureMap4, feature = model(imgs)
        # feature, output, local1, local2, local3, local4 = model(imgs)
        batch_time.update(time.time() - end)

        # Compute Loss
        # global_cls_loss_ = torch.nn.CrossEntropyLoss()(output, label)
        global_cls_loss_ = LabelSmoothLoss()(output, label)
        # local_cls_loss_ = LabelSmoothLoss()(local1, label) + LabelSmoothLoss()(local2, label) + LabelSmoothLoss()(
        #     local3, label) + LabelSmoothLoss()(local4, label)
        # score_loss_ = torch.mean(score)

        # loss_ = global_cls_loss_ + mask_loss_
        # loss_ = global_cls_loss_ + local_cls_loss_
        loss_ = global_cls_loss_

        # Back Propagation
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        # Compute accuracy, recall and loss
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Logs loss
        loss.update(float(loss_.cpu().data.item()))

        end = time.time()

    scheduler.step()

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

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f} Total Loss {loss:.4f}'''.format(
        acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)


def Test(args, model, test_dataloader, Best_Acc, epoch, writer, dataset):
    """Test."""
    global num

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_test_dataloader = iter(test_dataloader)

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                                 range(7)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, label, _) in enumerate(iter_test_dataloader):
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

    writer.add_scalar('Test_Recall_' + dataset, recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_' + dataset, acc_avg, epoch)

    LoggerInfo = '''
    [Test (Target Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)
    num = 0

    if acc_avg > Best_Acc and not args.isTest:
        Best_Acc = acc_avg
        print('[Save] Best Accuracy on %s: %.4f.' % (dataset, Best_Acc))

        if args.isSave:
            # Save Checkpoints
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(args.OutputPath,
                                                                   '{}_{}_{}.pkl'.format(args.sourceDataset,
                                                                                         dataset, args.Network)))
            else:
                torch.save(model.state_dict(), os.path.join(args.OutputPath,
                                                            '{}_{}_{}.pkl'.format(args.sourceDataset,
                                                                                  dataset, args.Network)))

    return Best_Acc


def main():
    """Main."""

    # Parse Argument
    args = parser.parse_args()
    if not args.isTest:
        sys.stdout = Logger(
            osp.join('./Logs/', '{}_{}.txt'.format(args.sourceDataset, args.Network)))
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
    print('Batch Size: %d' % args.batch_size)

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

    val_dataset_RAFDB = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    val_dataset_JAFFE = JAFFEDataSet(args.jaffe_path, transform=data_transforms_val)
    val_dataset_FER2013 = FER2013DataSet(args.fer2013_path, phase='test', transform=data_transforms_val)
    val_dataset_FER2013Plus = FER2013PlusDataSet(args.fer2013plus_path, phase='test', transform=data_transforms_val)
    val_dataset_SFEW = SFEWDataSet(args.sfew_path, phase='test', transform=data_transforms_val)
    val_dataset_ExpW = ExpWDataSet(args.expw_path, phase='test', transform=data_transforms_val)
    val_dataset_AffectNet = AffectNetDataSet(args.affectnet_path, phase='test', transform=data_transforms_val)

    print('Validation RAFDB set size:', val_dataset_RAFDB.__len__())
    print('The Validation Target dataset distribute:', val_dataset_RAFDB.__distribute__())
    print('Validation JAFFE set size:', val_dataset_JAFFE.__len__())
    print('The Validation Target dataset distribute:', val_dataset_JAFFE.__distribute__())
    print('Validation FER2013Plus set size:', val_dataset_FER2013Plus.__len__())
    print('The Validation Target dataset distribute:', val_dataset_FER2013Plus.__distribute__())
    print('Validation SFEW set size:', val_dataset_SFEW.__len__())
    print('The Validation Target dataset distribute:', val_dataset_SFEW.__distribute__())
    print('Validation ExpW set size:', val_dataset_ExpW.__len__())
    print('The Validation Target dataset distribute:', val_dataset_ExpW.__distribute__())
    print('Validation AffectNet set size:', val_dataset_AffectNet.__len__())
    print('The Validation Target dataset distribute:', val_dataset_AffectNet.__distribute__())

    val_loader_RAFDB = torch.utils.data.DataLoader(val_dataset_RAFDB,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False,
                                                   pin_memory=True)
    val_loader_JAFFE = torch.utils.data.DataLoader(val_dataset_JAFFE,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False,
                                                   pin_memory=True)
    val_loader_FER2013Plus = torch.utils.data.DataLoader(val_dataset_FER2013Plus,
                                                         batch_size=args.batch_size,
                                                         num_workers=args.workers,
                                                         shuffle=False,
                                                         pin_memory=True)
    val_loader_SFEW = torch.utils.data.DataLoader(val_dataset_SFEW,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.workers,
                                                  shuffle=False,
                                                  pin_memory=True)
    val_loader_ExpW = torch.utils.data.DataLoader(val_dataset_ExpW,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.workers,
                                                  shuffle=False,
                                                  pin_memory=True)
    val_loader_AffectNet = torch.utils.data.DataLoader(val_dataset_AffectNet,
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

    # Set Optimizer
    print('Buliding Optimizer...')
    params = model.parameters()
    optimizer = optim.AdamW(params, betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Acc_RAFDB, Best_Acc_JAFFE, Best_Acc_FER2013Plus, Best_Acc_SFEW, Best_Acc_ExpW, Best_Acc_AffectNet = 0, 0, 0, 0, 0, 0

    # Running Experiment
    print("Run Experiment...")

    writer = SummaryWriter(os.path.join('/home/zhongtao/code/CrossDomainFER/my_method/LogInfo',
                                        '{}_{}'.format(args.sourceDataset, args.Network)))

    for epoch in range(1, args.epochs + 1):

        if not args.isTest:
            Train(args, model, train_loader, optimizer, scheduler, epoch, writer)

        Best_Acc_RAFDB = Test(args, model, val_loader_RAFDB, Best_Acc_RAFDB, epoch, writer, 'RAFDB')
        Best_Acc_JAFFE = Test(args, model, val_loader_JAFFE, Best_Acc_JAFFE, epoch, writer, 'JAFFE')
        Best_Acc_FER2013Plus = Test(args, model, val_loader_FER2013Plus, Best_Acc_FER2013Plus, epoch, writer,
                                    'FER2013Plus')
        Best_Acc_SFEW = Test(args, model, val_loader_SFEW, Best_Acc_SFEW, epoch, writer, 'SFEW')
        Best_Acc_ExpW = Test(args, model, val_loader_ExpW, Best_Acc_ExpW, epoch, writer, 'ExpW')
        Best_Acc_AffectNet = Test(args, model, val_loader_AffectNet, Best_Acc_AffectNet, epoch, writer, 'AffectNet')

        # if args.showFeature:
        #     VisualizationForTwoDomain(
        #         '/home/zhongtao/code/CrossDomainFER/my_method/visualization/{}_{}_{:>02d}.pdf'.format(
        #             args.sourceDataset, args.targetDataset, epoch), model, train_loader, "RAFDB")

        torch.cuda.empty_cache()

    writer.close()
    print('Best Accuarcy on RAFDB:%.4f' % (Best_Acc_RAFDB))
    print('Best Accuarcy on JAFFE:%.4f' % (Best_Acc_JAFFE))
    print('Best Accuarcy on FER2013Plus:%.4f' % (Best_Acc_FER2013Plus))
    print('Best Accuarcy on SFEW:%.4f' % (Best_Acc_SFEW))
    print('Best Accuarcy on ExpW:%.4f' % (Best_Acc_ExpW))
    print('Best Accuarcy on AffectNet:%.4f' % (Best_Acc_AffectNet))


if __name__ == '__main__':
    main()
