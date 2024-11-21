import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Utils import LabelSmoothLoss, AverageMeter, Compute_Accuracy, Show_Accuracy, Similarity_Loss, show_feature_map, \
    calculate_CAM, make_histogram, BulidModel
from Datasets import JAFFEDataSet, RafDataSet, FER2013DataSet, ExpWDataSet, SFEWDataSet, AffectNetDataSet
from Model_pre import SANN

best_acc_source = 0
best_epoch_acc_source = 0
best_recall_source = 0
best_epoch_recall_source = 0
best_acc_target = 0
best_epoch_acc_target = 0
best_recall_target = 0
best_epoch_recall_target = 0
num_train = 0
num_test = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Log_Name', default='test', type=str, help='Logs Name')
    parser.add_argument('--DrawCAM', type=bool, default=False, help='Draw the CAM')
    parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50'])
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model',
                        default='/home/zhongtao/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    parser.add_argument('--OutputPath', default='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints', type=str,
                        help='Output Path')
    parser.add_argument('--sourceDataset', type=str, default='RAFDB',
                        choices=['RAFDB', 'SFEW', 'FER2013', 'AffectNet', 'ExpW'])
    parser.add_argument('--targetDataset', type=str, default='JAFFE',
                        choices=['RAFDB', 'SFEW', 'FER2013', 'AffectNet', 'JAFFE', 'ExpW'])
    parser.add_argument('--raf_path', type=str, default='/home/zhongtao/datasets/RAFDB',
                        help='Raf-DB dataset path.')
    parser.add_argument('--jaffe-path', type=str, default='/home/zhongtao/datasets/jaffedbase',
                        help='JAFFE dataset path.')
    parser.add_argument('--fer2013-path', type=str, default='/home/zhongtao/datasets/FER2013',
                        help='FER2013 dataset path.')
    parser.add_argument('--expw-path', type=str, default='/home/zhongtao/datasets/ExpW',
                        help='ExpW dataset path.')
    parser.add_argument('--sfew-path', type=str, default='/home/zhongtao/datasets/SFEW2.0',
                        help='SFEW dataset path.')
    parser.add_argument('--pretrained', type=str,
                        default='/home/zhongtao/code/Self-Cure-Network/models/vit_base_patch16_224.pth',
                        help='Pretrained weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--seed', type=int, default=2022, help='Set seed.')
    return parser.parse_args()


def run_training():
    args = parse_args()
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


    model = SANN(args, 18, num_classes=7)
    # model = BulidModel(args)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
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

    val_dataset_source = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    val_dataset_target = JAFFEDataSet(args.jaffe_path, transform=data_transforms_val)
    # val_dataset_target = FER2013DataSet(args.fer2013_path, phase='test', transform=data_transforms_val)
    # val_dataset_target = ExpWDataSet(args.expw_path, phase='test', transform=data_transforms_val)
    # val_dataset_target = SFEWDataSet(args.sfew_path, phase='test', transform=data_transforms_val)
    # val_dataset = AffectNetDataSet(args.affectnet_path, phase='test', transform=data_transforms_val)
    print('Validation Source set size:', val_dataset_source.__len__())
    print('Validation Target set size:', val_dataset_target.__len__())

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

    params = model.parameters()
    optimizer = optim.AdamW(params, betas=(0.9, 0.999), lr=0.0002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    # model = nn.DataParallel(model)
    model = model.cuda()

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothLoss()

    writer = SummaryWriter(os.path.join('/home/zhongtao/code/CrossDomainFER/my_method/LogInfo', args.Log_Name))

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        iter_cnt = 0

        acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], \
                            [AverageMeter() for i in range(7)]
        inter, intra = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

        model.train()

        for batch_i, (imgs, lable, indexes) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()

            feature, outputs = model(imgs)

            outputs = outputs.cuda()
            lable = lable.cuda()

            loss_ = criterion(outputs, lable)
            # loss_inter, loss_intra = Similarity_Loss(args, feature, lable, inter, intra)

            loss = loss_
            # loss = loss_ + loss_inter - loss_intra
            loss.backward()

            optimizer.step()

            running_loss += loss

            Compute_Accuracy(args, outputs, lable, acc, prec, recall)

        scheduler.step()

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, 7)

        running_loss = running_loss / iter_cnt

        LoggerInfo = '''[Train]:\nEpoch {0}\n    Learning Rate {1}\n'''.format(epoch, scheduler.get_lr())

        LoggerInfo += AccuracyInfo

        LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}'''.format(
            acc_avg, prec_avg, recall_avg, f1_avg)

        print('========================================================================================')
        print(LoggerInfo)

        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (epoch, acc_avg, running_loss))
        # print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. Inter similarity: %.3f. Intra similarity: %.3f\n' % (
        # i, accuarcy, running_loss, loss_inter, loss_intra))

        with torch.no_grad():
            global best_acc_source
            global best_epoch_acc_source
            global best_recall_source
            global best_epoch_recall_source
            global best_acc_target
            global best_epoch_acc_target
            global best_recall_target
            global best_epoch_recall_target
            global num_train
            global num_test
            running_loss_train = 0.0
            iter_cnt = 0
            num_train = 0
            num_test = 0

            acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], \
                                [AverageMeter() for i in range(7)]
            inter, intra = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

            model.eval()

            for batch_i, (imgs, lable, _) in enumerate(val_loader_source):
                imgs = imgs.cuda()
                feature, outputs = model(imgs)

                outputs = outputs.cuda()
                lable = lable.cuda()

                loss_ = criterion(outputs, lable)
                # loss_inter, loss_intra = Similarity_Loss(args, feature, lable, inter, intra)

                # loss = loss_ + loss_inter - loss_intra
                loss = loss_

                running_loss_train += loss
                iter_cnt += 1

                Compute_Accuracy(args, outputs, lable, acc, prec, recall)
                # show_feature_map(args, imgs, feature, num_train, 'train')
                if args.DrawCAM:
                    net_name = []
                    params = []
                    for name, param in model.named_parameters():
                        net_name.append(name)
                        params.append(param)
                    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
                    calculate_CAM(args, imgs, outputs, feature, weight_softmax, num_train, 'train', epoch)
                    # if epoch == 1:
                    #     make_histogram(args, feature, 'train', num_train, 'first')
                    # if epoch == args.epochs:
                    #     make_histogram(args, feature, 'train', num_train, 'last')
                    num_train += imgs.shape[0]

            AccuracyInfo, acc_avg_source, prec_avg_source, recall_avg_source, f1_avg_source = Show_Accuracy(acc, prec,
                                                                                                            recall, 7)

            writer.add_scalar('Accuracy Source', acc_avg_source, epoch)
            writer.add_scalar('Precision Source', prec_avg_source, epoch)
            writer.add_scalar('Recall Source', recall_avg_source, epoch)
            writer.add_scalar('F1 Source', f1_avg_source, epoch)

            running_loss_train = running_loss_train / iter_cnt
            writer.add_scalar('Loss Source', running_loss_train, epoch)

            LoggerInfo = '''[Test (Source Domain)]:\n    Learning Rate {0}\n'''.format(scheduler.get_lr())

            LoggerInfo += AccuracyInfo

            LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}'''.format(
                acc_avg_source, prec_avg_source, recall_avg_source, f1_avg_source)

            print(LoggerInfo)

            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (epoch, acc_avg_source, running_loss_train))
            # print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f Inter similarity: %.3f Intra similarity: %.3f\n' % (
            # i, accuarcy, running_loss, loss_inter, loss_intra))

            if acc_avg_source > best_acc_source:
                best_acc_source = acc_avg_source
                best_epoch_acc_source = epoch

                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Accuracy_source.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                  args.targetDataset)))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Accuracy_source.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                  args.targetDataset)))
            if recall_avg_source > best_recall_source:
                best_recall_source = recall_avg_source
                best_epoch_recall_source = epoch

                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Recall_source.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                args.targetDataset)))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Recall_source.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                args.targetDataset)))

            # test target
            running_loss_test = 0.0
            iter_cnt = 0

            acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], \
                                [AverageMeter() for i in range(7)]
            inter, intra = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

            for batch_i, (imgs, lable, _) in enumerate(val_loader_target):
                imgs = imgs.cuda()
                feature, outputs = model(imgs)

                outputs = outputs.cuda()
                lable = lable.cuda()

                loss_ = criterion(outputs, lable)
                # loss_inter, loss_intra = Similarity_Loss(args, feature, lable, inter, intra)

                loss = loss_
                # loss = loss_ + loss_inter - loss_intra

                running_loss_test += loss
                iter_cnt += 1

                Compute_Accuracy(args, outputs, lable, acc, prec, recall)
                # show_feature_map(args, imgs, feature, num_test, 'test')
                if args.DrawCAM:
                    net_name = []
                    params = []
                    for name, param in model.named_parameters():
                        net_name.append(name)
                        params.append(param)
                    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
                    calculate_CAM(args, imgs, outputs, feature, weight_softmax, num_test, 'test', epoch)
                    # if epoch == 1:
                    #     make_histogram(args, feature, 'test', num_test, 'first')
                    # if epoch == args.epochs:
                    #     make_histogram(args, feature, 'test', num_test, 'last')
                    num_test += imgs.shape[0]

            AccuracyInfo, acc_avg_target, prec_avg_target, recall_avg_target, f1_avg_target = Show_Accuracy(acc, prec,
                                                                                                            recall, 7)
            writer.add_scalar('Accuracy Target', acc_avg_target, epoch)
            writer.add_scalar('Precision Target', prec_avg_target, epoch)
            writer.add_scalar('Recall Target', recall_avg_target, epoch)
            writer.add_scalar('F1 Target', f1_avg_target, epoch)

            running_loss_test = running_loss_test / iter_cnt
            writer.add_scalar('Loss Target', running_loss_test, epoch)

            LoggerInfo = '''[Test (Target Domain)]:\n    Learning Rate {0}\n'''.format(scheduler.get_lr())

            LoggerInfo += AccuracyInfo

            LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}'''.format(
                acc_avg_target, prec_avg_target, recall_avg_target, f1_avg_target)

            print(LoggerInfo)

            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (epoch, acc_avg_target, running_loss_test))
            # print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f Inter similarity: %.3f Intra similarity: %.3f\n' % (
            # i, accuarcy, running_loss, loss_inter, loss_intra))

            if acc_avg_target > best_acc_target:
                best_acc_target = acc_avg_target
                best_epoch_acc_target = epoch

                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Accuracy_target.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                  args.targetDataset)))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Accuracy_target.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                  args.targetDataset)))

            if recall_avg_target > best_recall_target:
                best_recall_target = recall_avg_target
                best_epoch_recall_target = epoch

                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Recall_target.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                args.targetDataset)))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(args.OutputPath,
                                            '{}_{}_{}_Recall_target.pkl'.format('TrainOnSource', args.sourceDataset,
                                                                                args.targetDataset)))


if __name__ == "__main__":
    run_training()
    print("[Epoch %d] Best Validation Source accuracy:%.4f." % (best_epoch_acc_source, best_acc_source))
    print("[Epoch %d] Best Validation Source recall:%.4f." % (best_epoch_recall_source, best_recall_source))
    print("[Epoch %d] Best Validation Target accuracy:%.4f." % (best_epoch_acc_target, best_acc_target))
    print("[Epoch %d] Best Validation Target recall:%.4f." % (best_epoch_recall_target, best_recall_target))
