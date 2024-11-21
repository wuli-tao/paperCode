import os
import random
import sys

import matplotlib.pyplot as plt

sys.path.append('.')
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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser(description='Expression Classification Training')

parser.add_argument('--Log_Name', type=str, default='test', help='Logs Name')
parser.add_argument('--OutputPath', default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints', type=str,
                    help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Network', type=str, default='AConv+DHSA+global(x)',
                    choices=['Baseline', 'AConv', 'AConv+DHSA', 'AConv+DHSA+global', 'AConv+DHSA+global(x)', 'unified'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model',
                    default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB+FER2013Plus+SFEW_AffectNet_AConv+DHSA+global(x).pkl')
parser.add_argument('--GPU_ID', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=224, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAFDB',
                    choices=['RAFDB', 'SFEW', 'FER2013', 'FER2013Plus', 'AffectNet', 'ExpW'])
parser.add_argument('--targetDataset', type=str, default='RAFDB',
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
        transforms.ToTensor(), ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])])
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

    target_layers0 = [model.conv4.conv1, model.conv4.spatial1[3]]
    target_layers1 = [model.conv4.conv2, model.conv4.spatial2[3]]
    target_layers2 = [model.conv4.conv3, model.conv4.spatial3[3]]
    target_layers3 = [model.conv4.conv4, model.conv4.spatial4[3]]

    cam0 = GradCAM(model=model, target_layers=target_layers0, use_cuda=True)
    cam1 = GradCAM(model=model, target_layers=target_layers1, use_cuda=True)
    cam2 = GradCAM(model=model, target_layers=target_layers2, use_cuda=True)
    cam3 = GradCAM(model=model, target_layers=target_layers3, use_cuda=True)

    for step, (img, label, _) in enumerate(val_loader_target):
        print(step)
        img_ = img.squeeze().permute(1, 2, 0)
        input_tensor = torch.Tensor(img.T)
        grayscale_cam0 = cam0(input_tensor=img, targets=[ClassifierOutputTarget(6)])
        grayscale_cam1 = cam1(input_tensor=img, targets=[ClassifierOutputTarget(6)])
        grayscale_cam2 = cam2(input_tensor=img, targets=[ClassifierOutputTarget(6)])
        grayscale_cam3 = cam3(input_tensor=img, targets=[ClassifierOutputTarget(6)])

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam0 = grayscale_cam0[0, :]
        grayscale_cam1 = grayscale_cam1[0, :]
        grayscale_cam2 = grayscale_cam2[0, :]
        grayscale_cam3 = grayscale_cam3[0, :]

        # print(grayscale_cam)
        visualization0, d0, b0 = show_cam_on_image(img_, grayscale_cam0, use_rgb=True)
        visualization1, d1, b1 = show_cam_on_image(img_, grayscale_cam1, use_rgb=True)
        visualization2, d2, b2 = show_cam_on_image(img_, grayscale_cam2, use_rgb=True)
        visualization3, d3, b3 = show_cam_on_image(img_, grayscale_cam3, use_rgb=True)

        # plt.subplot(331, label='orignal')
        # plt.imshow(img_)
        # plt.subplot(332)
        # plt.imshow(b0)
        # plt.subplot(333)
        # plt.imshow(b1)
        # plt.subplot(334)
        # plt.imshow(b2)
        # plt.subplot(335)
        # plt.imshow(b3)
        #
        # plt.savefig(
        #     fname='/home/zhongtao/code/CrossDomainFER/my_method/CAM/{}/heatmap{}.png'.format(
        #         args.targetDataset, step), format='jpg', bbox_inches='tight')
        plt.plot()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去除x轴
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去除y轴
        plt.imshow(b0)
        plt.savefig(fname='/home/zhongtao/code/CrossDomainFER/my_method/CAM/{}/heatmapA{}_1.png'.format(
                args.targetDataset, step), format='jpg', bbox_inches='tight')
        plt.clf()
        plt.plot()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去除x轴
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去除y轴
        plt.imshow(b1)
        plt.savefig(fname='/home/zhongtao/code/CrossDomainFER/my_method/CAM/{}/heatmapA{}_2.png'.format(
            args.targetDataset, step), format='jpg', bbox_inches='tight')
        plt.clf()
        plt.plot()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去除x轴
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去除y轴
        plt.imshow(b2)
        plt.savefig(fname='/home/zhongtao/code/CrossDomainFER/my_method/CAM/{}/heatmapA{}_3.png'.format(
            args.targetDataset, step), format='jpg', bbox_inches='tight')
        plt.clf()
        plt.plot()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去除x轴
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去除y轴
        plt.imshow(b3)
        plt.savefig(fname='/home/zhongtao/code/CrossDomainFER/my_method/CAM/{}/heatmapA{}_4.png'.format(
            args.targetDataset, step), format='jpg', bbox_inches='tight')
        plt.clf()
        # fileName='note.txt'
        # with open(fileName,'w',encoding='utf-8')as file:
        #     file.write(str(d0))
        #     file.write(str(d1))
        #     file.write(str(d2))
        #     file.write(str(d3))
        #     file.write(str(d4))
        #     file.write(str(d5))
        # plt.savefig(
        #     fname='/home/zhongtao/code/CrossDomainFER/my_method/CAM/{}/CAM{}.png'.format(
        #         args.targetDataset, step), format='jpg', bbox_inches='tight')
        # plt.figure("dog")
        # plt.imshow(visualization)
        # plt.show()
        # plt.savefig(fname='8_.pdf', format="pdf", bbox_inches='tight')
        # exit()


if __name__ == '__main__':
    main()
