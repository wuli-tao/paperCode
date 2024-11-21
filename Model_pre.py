from functools import partial
from random import randint

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
import torch.nn.functional as F
from collections import namedtuple

from ViT import vit_base_patch16_224


class SANN_fus(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(SANN_fus, self).__init__()
        self.drop_rate = drop_rate

        if NumOfLayer == 18:
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        # self.vit = load_vit(args)

        self.pre_conv = nn.Sequential(*list(resnet.children())[0:4]).cuda()
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5]).cuda()
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6]).cuda()
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7]).cuda()
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8]).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool2d((14, 14))
        self.upsample = nn.UpsamplingNearest2d((14, 14))

        # 动态卷积
        # self.conv1 = nn.Conv2d(64, 768, 3, 1, 1)
        # self.conv2 = nn.Conv2d(128, 768, 3, 1, 1)
        # self.conv3 = nn.Conv2d(256, 768, 3, 1, 1)
        # self.conv4 = nn.Conv2d(512, 768, 3, 1, 1)
        # self.conv1 = SKAttention(64, 768 // 4, 1)
        # self.conv2 = SKAttention(128, 768 // 4, 1)
        # self.conv3 = SKAttention(256, 768 // 4, 1)
        # self.conv4 = SKAttention(512, 768 // 4, 1)
        self.conv1 = Attention1(args, 64, 4)
        self.conv2 = Attention1(args, 128, 4)
        self.conv3 = Attention1(args, 256, 4)
        self.conv4 = Attention1(args, 512, 4)

        # self.SA1 = nn.Sequential(self.vit.blocks[0:1])
        # self.SA2 = nn.Sequential(self.vit.blocks[1:3])
        # self.SA3 = nn.Sequential(self.vit.blocks[3:7])
        # self.SA4 = nn.Sequential(self.vit.blocks[7:12])
        #
        # # 位置编码
        # self.pos_embed1 = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        # self.pos_embed2 = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        # self.pos_embed3 = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        # self.pos_embed4 = nn.Parameter(torch.zeros(1, 14 * 14, 768))

        # 对每一个14x14做阈值
        self.threshold = nn.Sequential(
            nn.Linear(3072, 768),
            # nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 3072),
            # nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(768 * 4, num_classes)
        self.fc1 = nn.Linear(768, 7)
        self.fc2 = nn.Linear(768, 7)
        self.fc3 = nn.Linear(768, 7)
        self.fc4 = nn.Linear(768, 7)
        self.bn = nn.BatchNorm2d(768)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, score1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        # featureMap1 = featureMap1.view(bs, c, -1)
        # featureMap1 = featureMap1.permute(0, 2, 1)
        # featureMap1 = featureMap1 + self.pos_embed1
        # featureMap1 = self.SA1(featureMap1)
        # featureMap1 = featureMap1.permute(0, 2, 1)
        # featureMap1 = featureMap1.view(bs, c, h, w)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, score2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        # featureMap2 = featureMap2.view(bs, c, -1)
        # featureMap2 = featureMap2.permute(0, 2, 1)
        # featureMap2 = featureMap2 + self.pos_embed2
        # featureMap2 = self.SA2(featureMap2)
        # featureMap2 = featureMap2.permute(0, 2, 1)
        # featureMap2 = featureMap2.view(bs, c, h, w)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, score3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        # featureMap3 = featureMap3.view(bs, c, -1)
        # featureMap3 = featureMap3.permute(0, 2, 1)
        # featureMap3 = featureMap3 + self.pos_embed3
        # featureMap3 = self.SA3(featureMap3)
        # featureMap3 = featureMap3.permute(0, 2, 1)
        # featureMap3 = featureMap3.view(bs, c, h, w)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, score4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        # featureMap4 = featureMap4.view(bs, c, -1)
        # featureMap4 = featureMap4.permute(0, 2, 1)
        # featureMap4 = featureMap4 + self.pos_embed4
        # featureMap4 = self.SA4(featureMap4)
        # featureMap4 = featureMap4.permute(0, 2, 1)
        # featureMap4 = featureMap4.view(bs, c, h, w)

        featureMap = torch.cat([featureMap1, featureMap2, featureMap3, featureMap4], dim=1)
        score = score1 + score2 + score3 + score4

        # 14x14阈值dropout
        # important = self.avgpool(featureMap)
        # important = important.view(important.size(0), -1)
        # threshold_ = self.threshold(important)
        # # mask = (important > threshold_).type(torch.uint8)
        # # print(torch.sum(mask))
        # th = torch.mean(important)
        # threshold = threshold_ - 0.5
        # mask = (threshold > 0).type(torch.uint8)
        # mask = mask.unsqueeze(2).unsqueeze(3)
        # featureMap = featureMap * mask

        feature = self.avgpool(featureMap)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        # loc1 = self.fc1(self.avgpool(featureMap1).view(bs, -1))
        # loc2 = self.fc2(self.avgpool(featureMap2).view(bs, -1))
        # loc3 = self.fc3(self.avgpool(featureMap3).view(bs, -1))
        # loc4 = self.fc4(self.avgpool(featureMap4).view(bs, -1))

        return featureMap, out, score
        # return featureMap, out, loc1, loc2, loc3, loc4


class SANN_select(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(SANN_select, self).__init__()
        self.drop_rate = drop_rate

        if NumOfLayer == 18:
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)

        self.pre_conv = nn.Sequential(*list(resnet.children())[0:4]).cuda()
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5]).cuda()
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6]).cuda()
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7]).cuda()
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8]).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool2d((14, 14))
        self.upsample = nn.UpsamplingNearest2d((14, 14))

        # 动态卷积
        # self.conv1 = nn.Conv2d(64, 768, 3, 1, 1)
        # self.conv2 = nn.Conv2d(128, 768, 3, 1, 1)
        # self.conv3 = nn.Conv2d(256, 768, 3, 1, 1)
        # self.conv4 = nn.Conv2d(512, 768, 3, 1, 1)
        # self.conv1 = SKAttention(64, 768 // 4, 1)
        # self.conv2 = SKAttention(128, 768 // 4, 1)
        # self.conv3 = SKAttention(256, 768 // 4, 1)
        # self.conv4 = SKAttention(512, 768 // 4, 1)
        self.conv1 = Attention(args, 64, 4, 0.7)
        self.conv2 = Attention(args, 128, 4, 0.8)
        self.conv3 = Attention(args, 256, 4, 0.9)
        self.conv4 = Attention(args, 512, 4, 1)

        # 对每一个14x14做阈值
        # self.threshold = nn.Sequential(
        #     nn.Linear(768, 768 // 4),
        #     # nn.BatchNorm1d(768),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(768 // 4, 768),
        #     # nn.BatchNorm1d(3072),
        #     nn.ReLU(inplace=True),
        #     nn.Sigmoid()
        # )

        self.select1 = nn.Sequential(
            nn.Linear(768, 768 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 4, 768),
            nn.Sigmoid()
        )
        self.select2 = nn.Sequential(
            nn.Linear(768, 768 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 4, 768),
            nn.Sigmoid()
        )
        self.select3 = nn.Sequential(
            nn.Linear(768, 768 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 4, 768),
            nn.Sigmoid()
        )
        self.select4 = nn.Sequential(
            nn.Linear(768, 768 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 4, 768),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(768, num_classes)
        self.bn = nn.BatchNorm2d(768)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, score1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        # featureMap1 = featureMap1.view(bs, c, -1)
        # featureMap1 = featureMap1.permute(0, 2, 1)
        # featureMap1 = featureMap1 + self.pos_embed1
        # featureMap1 = self.SA1(featureMap1)
        # featureMap1 = featureMap1.permute(0, 2, 1)
        # featureMap1 = featureMap1.view(bs, c, h, w)
        importance1 = self.select1(self.avgpool(featureMap1).view(bs, -1)).unsqueeze(2)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, score2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        # featureMap2 = featureMap2.view(bs, c, -1)
        # featureMap2 = featureMap2.permute(0, 2, 1)
        # featureMap2 = featureMap2 + self.pos_embed2
        # featureMap2 = self.SA2(featureMap2)
        # featureMap2 = featureMap2.permute(0, 2, 1)
        # featureMap2 = featureMap2.view(bs, c, h, w)
        importance2 = self.select2(self.avgpool(featureMap2).view(bs, -1)).unsqueeze(2)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, score3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        # featureMap3 = featureMap3.view(bs, c, -1)
        # featureMap3 = featureMap3.permute(0, 2, 1)
        # featureMap3 = featureMap3 + self.pos_embed3
        # featureMap3 = self.SA3(featureMap3)
        # featureMap3 = featureMap3.permute(0, 2, 1)
        # featureMap3 = featureMap3.view(bs, c, h, w)
        importance3 = self.select3(self.avgpool(featureMap3).view(bs, -1)).unsqueeze(2)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, score4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        # featureMap4 = featureMap4.view(bs, c, -1)
        # featureMap4 = featureMap4.permute(0, 2, 1)
        # featureMap4 = featureMap4 + self.pos_embed4
        # featureMap4 = self.SA4(featureMap4)
        # featureMap4 = featureMap4.permute(0, 2, 1)
        # featureMap4 = featureMap4.view(bs, c, h, w)
        importance4 = self.select4(self.avgpool(featureMap4).view(bs, -1)).unsqueeze(2)

        importance_vector = [importance1, importance2, importance3, importance4]
        importance_vector = torch.cat(importance_vector, dim=2)
        importance_vector = F.softmax(importance_vector, dim=2)
        importance_vector = importance_vector.unsqueeze(3)

        for i in range(4):
            if i == 0:
                featureMap1 = featureMap1 * importance_vector[:, :, i, :].unsqueeze(3)
            elif i == 1:
                featureMap2 = featureMap2 * importance_vector[:, :, i, :].unsqueeze(3)
            elif i == 2:
                featureMap3 = featureMap3 * importance_vector[:, :, i, :].unsqueeze(3)
            else:
                featureMap4 = featureMap4 * importance_vector[:, :, i, :].unsqueeze(3)

        featureMap = featureMap1 + featureMap2 + featureMap3 + featureMap4
        score = score1 + score2 + score3 + score4

        # 14x14阈值dropout
        # important = self.avgpool(featureMap)
        # important = important.view(important.size(0), -1)
        # threshold_ = self.threshold(important)
        # # mask = (important > threshold_).type(torch.uint8)
        # # print(torch.sum(mask))
        # th = torch.mean(important)
        # threshold = threshold_ - th
        # mask = (threshold < 0).type(torch.uint8)
        # print(torch.sum(mask))
        # mask = mask.unsqueeze(2).unsqueeze(3)
        # featureMap = featureMap * mask

        feature = self.avgpool(featureMap)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return featureMap, out


class Attention(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(768, 768 // ratio, 1, 1, 0),
            nn.BatchNorm2d(768 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(768 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(768 // ratio, 768),
            nn.Sigmoid()
        )
        self.weight1 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.BatchNorm1d(768 // ratio),
            nn.Tanh(),
            nn.Linear(768 // ratio, 1),
        )
        self.spatial2 = nn.Sequential(
            nn.Conv2d(768, 768 // ratio, 1, 1, 0),
            # nn.BatchNorm2d(768 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(768 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(768 // ratio, 768),
            nn.Sigmoid()
        )
        self.weight2 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.BatchNorm1d(768 // ratio),
            nn.Tanh(),
            nn.Linear(768 // ratio, 1),
        )
        self.spatial3 = nn.Sequential(
            nn.Conv2d(768, 768 // ratio, 1, 1, 0),
            nn.BatchNorm2d(768 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(768 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(768 // ratio, 768),
            nn.Sigmoid()
        )
        self.weight3 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.BatchNorm1d(768 // ratio),
            nn.Tanh(),
            nn.Linear(768 // ratio, 1),
        )
        self.spatial4 = nn.Sequential(
            nn.Conv2d(768, 768 // ratio, 1, 1, 0),
            # nn.BatchNorm2d(768 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(768 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(768 // ratio, 768),
            nn.Sigmoid()
        )
        self.weight4 = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.BatchNorm1d(768 // ratio),
            nn.Tanh(),
            nn.Linear(768 // ratio, 1),
        )

        self.proportion = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveMaxPool2d(1),
        )
        self.linear1 = nn.Linear(768, 768 // 2)
        self.linear2 = nn.Linear(768 // 2, 1)
        self.linear3 = nn.Linear(768, 768 // 2)
        self.linear4 = nn.Linear(768 // 2, 1)

        self.SA1 = Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=14, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=768)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((7, 7))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.resize(x)

        feature_map1 = self.conv1(x)
        channel_att1 = self.avg(feature_map1)
        channel_att1 = self.channel1(channel_att1.squeeze(2).squeeze(2))
        spatial_att1 = self.spatial1(feature_map1)
        weight1 = self.avg(feature_map1)
        weight1 = self.weight1(weight1.squeeze(2).squeeze(2))

        feature_map2 = self.conv2(x)
        channel_att2 = self.avg(feature_map2)
        channel_att2 = self.channel2(channel_att2.squeeze(2).squeeze(2))
        spatial_att2 = self.spatial2(feature_map2)
        weight2 = self.avg(feature_map2)
        weight2 = self.weight2(weight2.squeeze(2).squeeze(2))

        feature_map3 = self.conv3(x)
        channel_att3 = self.avg(feature_map3)
        channel_att3 = self.channel3(channel_att3.squeeze(2).squeeze(2))
        spatial_att3 = self.spatial3(feature_map3)
        weight3 = self.avg(feature_map3)
        weight3 = self.weight3(weight3.squeeze(2).squeeze(2))

        feature_map4 = self.conv4(x)
        channel_att4 = self.avg(feature_map4)
        channel_att4 = self.channel4(channel_att4.squeeze(2).squeeze(2))
        spatial_att4 = self.spatial4(feature_map4)
        weight4 = self.avg(feature_map4)
        weight4 = self.weight4(weight4.squeeze(2).squeeze(2))

        # feature_map5 = self.conv5(x)
        # channel_att5 = self.avg(feature_map5)
        # channel_att5 = self.channel5(channel_att5.squeeze(2).squeeze(2))
        # spatial_att5 = self.spatial5(feature_map5)
        # weight5 = self.avg(feature_map5)
        # weight5 = self.weight5(weight5.squeeze(2).squeeze(2))
        # self_att5 = self.attention5(feature_map5)

        weight_vector = [weight1, weight2, weight3, weight4]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(4):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                # feature_map1 = self.resize(feature_map1)
            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                # feature_map2 = self.resize(feature_map2)
            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                # feature_map3 = self.resize(feature_map3)
            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                # feature_map4 = self.resize(feature_map4)

        # feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        temp1 = torch.cat((feature_map1, feature_map2), 2)
        temp2 = torch.cat((feature_map3, feature_map4), 2)
        feature_map = torch.cat((temp1, temp2), 3)

        # 续贯操作，SA+FA
        # bs, c, _, _ = feature_map.shape
        # feature_map_1 = feature_map.view(bs, c, -1)
        # feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.SA1(feature_map_1)
        # feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        # feature_map_1 = feature_map_1.view(bs, c, 14, 14)

        # 加权操作，SA，FA
        bs, c, _, _ = feature_map.shape
        feature_map_a1 = feature_map
        feature_map_a1 = feature_map_a1.view(bs, c, -1)
        feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        feature_map_a1, _ = self.SA1(feature_map_a1)
        feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        feature_map_a1 = feature_map_a1.view(bs, c, 14, 14)

        # feature_map_a2 = feature_map.view(bs, c, -1)
        # feature_map_a2 = feature_map_a2.permute(0, 2, 1)
        # feature_map_a2, _ = self.SA2(feature_map_a2)
        # feature_map_a2 = feature_map_a2.permute(0, 2, 1)
        # feature_map_a2 = feature_map_a2.view(bs, c, 14, 14)

        feature_map_a2 = feature_map
        feature_map_a2 = feature_map_a2.view(bs, c, -1)
        feature_map_a2, score = self.FA1(feature_map_a2)
        feature_map_a2 = feature_map_a2.view(bs, c, 14, 14)

        # 自适应两feature map比例
        # weight_f1 = self.proportion(feature_map_a1).squeeze(2).squeeze(2)
        # weight_f1 = self.linear1(weight_f1)
        # weight_f1 = self.linear2(weight_f1)
        # weight_f2 = self.proportion(feature_map_a2).squeeze(2).squeeze(2)
        # weight_f2 = self.linear1(weight_f2)
        # weight_f2 = self.linear2(weight_f2)
        #
        # vector = [weight_f1, weight_f2]
        # vector = torch.cat(vector, dim=1)
        # vector = self.softmax(vector)
        # vector = vector.unsqueeze(2).unsqueeze(3)
        #
        # for i in range(2):
        #     if i == 0:
        #         feature_map_a1 = feature_map_a1 * vector[:, i, :, :].unsqueeze(3)
        #     if i == 1:
        #         feature_map_a2 = feature_map_a2 * vector[:, i, :, :].unsqueeze(3)
        #
        # feature_map = feature_map_a1 + feature_map_a2

        feature_map = (feature_map_a1 + feature_map_a2) / 2

        # feature_map = (feature_map_1 + feature_map_2) / 2

        return feature_map, 0


class Attention1(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(Attention1, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 768, 3, 1, 1)
        self.spatial_1 = nn.Sequential(
            nn.Conv2d(768, 768 // ratio, 1, 1, 0),
            nn.BatchNorm2d(768 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(768 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.spatial_3 = nn.Sequential(
            nn.Conv2d(768, 768 // ratio, 3, 1, 1),
            nn.BatchNorm2d(768 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(768 // ratio, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.channel_avg = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(768 // ratio, 768),
            nn.Sigmoid()
        )
        self.channel_max = nn.Sequential(
            nn.Linear(768, 768 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(768 // ratio, 768),
            nn.Sigmoid()
        )

        self.SA1 = Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=14, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=768)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((7, 7))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.resize(x)
        feature_map1 = self.conv1(x)
        channel_avg = self.avg(feature_map1)
        channel_avg = self.channel_avg(channel_avg.squeeze(2).squeeze(2))
        feature_map1 = feature_map1 * channel_avg.unsqueeze(2).unsqueeze(3)

        feature_map2 = self.conv2(x)
        spatial_att1 = self.spatial_1(feature_map2)
        feature_map2 = feature_map2 * spatial_att1

        feature_map3 = self.conv3(x)
        channel_max = self.max(feature_map1)
        channel_max = self.channel_max(channel_max.squeeze(2).squeeze(2))
        feature_map3 = feature_map3 * channel_max.unsqueeze(2).unsqueeze(3)

        feature_map4 = self.conv4(x)
        spatial_att3 = self.spatial_3(feature_map4)
        feature_map4 = feature_map4 * spatial_att3

        temp1 = torch.cat((feature_map1, feature_map2), 2)
        temp2 = torch.cat((feature_map3, feature_map4), 2)
        feature_map = torch.cat((temp1, temp2), 3)

        # 续贯操作，SA+FA
        # bs, c, _, _ = feature_map.shape
        # feature_map_1 = feature_map.view(bs, c, -1)
        # feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.SA1(feature_map_1)
        # feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        # feature_map = feature_map_1.view(bs, c, 14, 14)

        # 加权操作，SA，FA
        bs, c, _, _ = feature_map.shape
        feature_map_a1 = feature_map
        feature_map_a1 = feature_map_a1.view(bs, c, -1)
        feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        feature_map_a1, _ = self.SA1(feature_map_a1)
        feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        feature_map_a1 = feature_map_a1.view(bs, c, 14, 14)

        # feature_map_a2 = feature_map
        # feature_map_a2 = feature_map_a2.view(bs, c, -1)
        # feature_map_a2 = feature_map_a2.permute(0, 2, 1)
        # feature_map_a2, _ = self.SA2(feature_map_a2)
        # feature_map_a2 = feature_map_a2.permute(0, 2, 1)
        # feature_map_a2 = feature_map_a2.view(bs, c, 14, 14)

        feature_map_a2 = feature_map
        feature_map_a2 = feature_map_a2.view(bs, c, -1)
        feature_map_a2, score = self.FA1(feature_map_a2)
        feature_map_a2 = feature_map_a2.view(bs, c, 14, 14)

        feature_map = (feature_map_a1 + feature_map_a2) / 2

        # feature_map = (feature_map_1 + feature_map_2) / 2

        return feature_map, 0


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SAttention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 head_drop_ratio=0.,
                 num_patches=196):
        super(SAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.head_drop_ratio = head_drop_ratio
        # self.bias = nn.Parameter(torch.randn(1, num_heads, num_patches, num_patches))
        # nn.init.kaiming_uniform_(self.bias, mode='fan_in')

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        '''
        在这里开始丢头
        '''
        # prob = torch.distributions.bernoulli.Bernoulli(self.head_drop_ratio)
        # for i in range(self.num_heads):
        #     attn[:, i, :, :] = 1 * prob.sample() * attn[:, i, :, :]
        #     print(attn[:, i, :, :])

        prob = torch.zeros(1, self.num_heads) + self.head_drop_ratio
        prob = torch.bernoulli(prob)
        prob = prob.unsqueeze(2).unsqueeze(3).cuda()
        attn = attn * prob

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = self.attn_drop(attn) + self.bias

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 head_drop_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_patches=196):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                               head_drop_ratio=head_drop_ratio, num_patches=num_patches)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        temp, score = self.attn(self.norm1(x))
        x = x + self.drop_path(temp)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, score


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, padding=1, shape=0, relative=False,
                 stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


def load_vit(args):
    model_vit = vit_base_patch16_224()
    pretrained = torch.load(args.pretrained)
    model_state_dict = model_vit.state_dict()
    loaded_keys = 0
    total_keys = 0
    for key in pretrained:
        if ((key == 'head.weight') | (key == 'head.bias')
                | (key == 'patch_embed.proj.weight') | (key == 'patch_embed.proj.bias')
        ):
            pass
        else:
            model_state_dict[key] = pretrained[key]
            total_keys += 1
            if key in model_state_dict:
                loaded_keys += 1
    print("Loaded params_vit num:", loaded_keys)
    print("Total params_vit num:", total_keys)
    model_vit.load_state_dict(model_state_dict, strict=True)

    return model_vit


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Backbone_onlyGlobal(nn.Module):
    def __init__(self, args, numOfLayer, num_classes=7):
        super(Backbone_onlyGlobal, self).__init__()

        unit_module = bottleneck_IR

        self.input_layer = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            BatchNorm2d(64),
            PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[0]])  # get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[1]])  # get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[2]])  # get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[3]])  # get_block(in_channel=256, depth=512, num_units=3)])

        self.reshape = Reshape(args)
        self.mask = Mask()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768 * 4, num_classes)

        self.fc.apply(init_weights)

    def forward(self, x):
        featuremap = self.input_layer(x)

        featuremap1 = self.layer1(featuremap)  # Batch * 64 * 56 * 56
        featuremap2 = self.layer2(featuremap1)  # Batch * 128 * 28 * 28
        featuremap3 = self.layer3(featuremap2)  # Batch * 256 * 14 * 14
        featuremap4 = self.layer4(featuremap3)  # Batch * 512 * 7 * 7

        # featureMap = self.mask(featureMap)

        feature = self.avgpool(featuremap4)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        # return featureMap1, featureMap2, featureMap3, featureMap4, out
        return out, featuremap4


class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.vit = load_vit(args)

        # self.conv1 = nn.Conv2d(64, 768, 3, 1, 1)
        # self.conv2 = nn.Conv2d(128, 768, 3, 1, 1)
        # self.conv3 = nn.Conv2d(256, 768, 3, 1, 1)
        # self.conv4 = nn.Conv2d(512, 768, 3, 1, 1)
        # self.conv1 = SKAttention(64, 768 // 4, 1)
        # self.conv2 = SKAttention(128, 768 // 4, 1)
        # self.conv3 = SKAttention(256, 768 // 4, 1)
        # self.conv4 = SKAttention(512, 768 // 4, 1)
        self.conv1 = Attention(64, 4)
        self.conv2 = Attention(128, 4)
        self.conv3 = Attention(256, 4)
        self.conv4 = Attention(512, 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d((14, 14))
        self.avgpool2 = nn.AdaptiveAvgPool2d((14, 14))
        self.avgpool3 = nn.AdaptiveAvgPool2d((14, 14))
        # self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=2)

        # 位置编码
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, 14 * 14, 768))

        self.SA1 = nn.Sequential(self.vit.blocks[0:1])
        self.SA2 = nn.Sequential(self.vit.blocks[1:3])
        self.SA3 = nn.Sequential(self.vit.blocks[3:7])
        self.SA4 = nn.Sequential(self.vit.blocks[7:12])

    def forward(self, featureMap1, featureMap2, featureMap3, featureMap4):
        bs, c, h, w = featureMap1.shape[0], 768, 14, 14

        featureMap1 = self.avgpool1(featureMap1)
        featureMap1 = self.conv1(featureMap1)
        featureMap1 = featureMap1.view(bs, c, -1)
        featureMap1 = featureMap1.permute(0, 2, 1)
        featureMap1 = featureMap1 + self.pos_embed1
        featureMap1 = self.SA1(featureMap1)
        featureMap1 = featureMap1.permute(0, 2, 1)
        featureMap1 = featureMap1.view(bs, c, h, w)

        featureMap2 = self.avgpool2(featureMap2)
        featureMap2 = self.conv2(featureMap2)
        featureMap2 = featureMap2.view(bs, c, -1)
        featureMap2 = featureMap2.permute(0, 2, 1)
        featureMap2 = featureMap2 + self.pos_embed2
        featureMap2 = self.SA2(featureMap2)
        featureMap2 = featureMap2.permute(0, 2, 1)
        featureMap2 = featureMap2.view(bs, c, h, w)

        featureMap3 = self.avgpool3(featureMap3)
        featureMap3 = self.conv3(featureMap3)
        featureMap3 = featureMap3.view(bs, c, -1)
        featureMap3 = featureMap3.permute(0, 2, 1)
        featureMap3 = featureMap3 + self.pos_embed3
        featureMap3 = self.SA3(featureMap3)
        featureMap3 = featureMap3.permute(0, 2, 1)
        featureMap3 = featureMap3.view(bs, c, h, w)

        # featureMap4 = self.upsample4(featureMap4)
        featureMap4 = self.conv4(featureMap4)
        featureMap4 = featureMap4.view(bs, c, -1)
        featureMap4 = featureMap4.permute(0, 2, 1)
        featureMap4 = featureMap4 + self.pos_embed4
        featureMap4 = self.SA4(featureMap4)
        featureMap4 = featureMap4.permute(0, 2, 1)
        featureMap4 = featureMap4.view(bs, c, h, w)

        return featureMap1, featureMap2, featureMap3, featureMap4


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        # 对每一个14x14做阈值
        self.threshold = nn.Sequential(
            nn.Linear(3072, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, featureMap):
        # 14x14阈值dropout
        important = self.avgpool(featureMap)
        important = important.view(important.size(0), -1)
        threshold = self.threshold(important)
        mask = (important > threshold).type(torch.uint8)
        mask = mask.unsqueeze(2).unsqueeze(3)
        featureMap = featureMap * mask

        return featureMap
