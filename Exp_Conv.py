from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
from torch.nn import Module, Parameter
import Utils


class AConv_DHSA_global_x(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(AConv_DHSA_global_x, self).__init__()
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
        self.conv1 = DynamicConv(args, 64, 4, 0.7)
        self.conv2 = DynamicConv(args, 128, 4, 0.8)
        self.conv3 = DynamicConv(args, 256, 4, 0.9)
        self.conv4 = DynamicConv(args, 512, 4, 1)

        self.select1 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select2 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select3 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select4 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(1024, num_classes)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = self.fc.weight.size(1)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, f1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        featuremap1 = featureMap1
        importance1 = self.select1(self.avgpool(featureMap1).view(bs, -1)).unsqueeze(2)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, f2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        featuremap2 = featureMap2
        importance2 = self.select2(self.avgpool(featureMap2).view(bs, -1)).unsqueeze(2)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, f3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        featuremap3 = featureMap3
        importance3 = self.select3(self.avgpool(featureMap3).view(bs, -1)).unsqueeze(2)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, f4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        featuremap4 = featureMap4
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

        featuremap = featureMap1 + featureMap2 + featureMap3 + featureMap4

        featureMap = self.avgpool(featuremap4)
        x = self.avgpool(x)
        feature = torch.cat((featureMap, x), dim=1)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return out, featuremap, featureMap1, featureMap2, featureMap3, featureMap4, feature

    def output_num(self):
        return self.output_dim


class DynamicConv(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial3 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial4 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )
        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.FA2 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((14, 14))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        if x.size(2) == 7:
            x = self.upsample(x)
        else:
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

        weight_vector = [weight1, weight2, weight3, weight4]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(4):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)

        feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        # Utils.VisualizationForDistrubution(self.avg(x).view(bs, -1), self.avg(feature_map1).view(bs, -1),
        #                                    self.avg(feature_map2).view(bs, -1), self.avg(feature_map3).view(bs, -1),
        #                                    self.avg(feature_map4).view(bs, -1))
        # Utils.VisualizationForDistrubution(self.avg(feature_map).view(bs, -1))

        # distribution dropout
        bs, c, _, _ = feature_map.shape
        feature_map_1 = feature_map.view(bs, c, -1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        Utils.AttentionHeatMap(feature_map_1, "in")
        feature_map_1, f = self.SA1(feature_map_1)
        Utils.AttentionHeatMap(feature_map_1, "out")
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        feature_map_1 = feature_map_1.view(bs, c, 14, 14)

        feature_map = feature_map_1 + feature_map

        return feature_map, f


class DynamicConv9(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv9, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial3 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial4 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial5 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel5 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial6 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel6 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial7 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel7 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial8 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel8 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial9 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel9 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight5 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight6 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight7 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight8 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight9 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=441)
        self.SA2 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.FA2 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((7, 7))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

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

        feature_map5 = self.conv5(x)
        channel_att5 = self.avg(feature_map5)
        channel_att5 = self.channel5(channel_att5.squeeze(2).squeeze(2))
        spatial_att5 = self.spatial5(feature_map5)
        weight5 = self.avg(feature_map5)
        weight5 = self.weight5(weight5.squeeze(2).squeeze(2))

        feature_map6 = self.conv6(x)
        channel_att6 = self.avg(feature_map6)
        channel_att6 = self.channel6(channel_att6.squeeze(2).squeeze(2))
        spatial_att6 = self.spatial6(feature_map6)
        weight6 = self.avg(feature_map6)
        weight6 = self.weight6(weight6.squeeze(2).squeeze(2))

        feature_map7 = self.conv7(x)
        channel_att7 = self.avg(feature_map7)
        channel_att7 = self.channel7(channel_att7.squeeze(2).squeeze(2))
        spatial_att7 = self.spatial7(feature_map7)
        weight7 = self.avg(feature_map7)
        weight7 = self.weight7(weight7.squeeze(2).squeeze(2))

        feature_map8 = self.conv8(x)
        channel_att8 = self.avg(feature_map8)
        channel_att8 = self.channel8(channel_att8.squeeze(2).squeeze(2))
        spatial_att8 = self.spatial8(feature_map8)
        weight8 = self.avg(feature_map8)
        weight8 = self.weight8(weight8.squeeze(2).squeeze(2))

        feature_map9 = self.conv9(x)
        channel_att9 = self.avg(feature_map9)
        channel_att9 = self.channel9(channel_att9.squeeze(2).squeeze(2))
        spatial_att9 = self.spatial9(feature_map9)
        weight9 = self.avg(feature_map9)
        weight9 = self.weight9(weight9.squeeze(2).squeeze(2))

        weight_vector = [weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(9):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 4:
                feature_map5 = feature_map5 * (channel_att5.unsqueeze(2).unsqueeze(3) + spatial_att5) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 5:
                feature_map6 = feature_map6 * (channel_att6.unsqueeze(2).unsqueeze(3) + spatial_att6) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 6:
                feature_map7 = feature_map7 * (channel_att7.unsqueeze(2).unsqueeze(3) + spatial_att7) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 7:
                feature_map8 = feature_map8 * (channel_att8.unsqueeze(2).unsqueeze(3) + spatial_att8) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 8:
                feature_map9 = feature_map9 * (channel_att9.unsqueeze(2).unsqueeze(3) + spatial_att9) \
                               * weight_vector[:, i, :, :].unsqueeze(3)

        # feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        temp1 = torch.cat((feature_map1, feature_map2, feature_map3), 2)
        temp2 = torch.cat((feature_map4, feature_map5, feature_map6), 2)
        temp3 = torch.cat((feature_map7, feature_map8, feature_map9), 2)
        feature_map = torch.cat((temp1, temp2, temp3), 3)
        # feature_map_global = feature_map1 + feature_map2 + feature_map3 + feature_map4
        # feature_map_global = self.upsample(feature_map_global)

        # 串联，SA+FA
        bs, c, _, _ = feature_map.shape
        feature_map_1 = feature_map.view(bs, c, -1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        feature_map_1, _ = self.SA1(feature_map_1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        feature_map_1 = feature_map_1.view(bs, c, 21, 21)

        # 加权操作，SA，FA
        # bs, c, _, _ = feature_map.shape
        # feature_map_a1 = feature_map
        # feature_map_a1 = feature_map_a1.view(bs, c, -1)
        # feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        # feature_map_a1, _ = self.SA2(feature_map_a1)
        # feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        # feature_map_a1 = feature_map_a1.view(bs, c, 14, 14)
        #
        # feature_map_a2 = feature_map
        # feature_map_a2 = feature_map_a2.view(bs, c, -1)
        # feature_map_a2, score = self.FA2(feature_map_a2)
        # feature_map_a2 = feature_map_a2.view(bs, c, 14, 14)
        #
        # weight_a1 = self.avg(feature_map_a1).squeeze(2).squeeze(2)
        # weight_a1 = self.linear1(weight_a1)
        # weight_a2 = self.avg(feature_map_a2).squeeze(2).squeeze(2)
        # weight_a2 = self.linear2(weight_a2)
        #
        # vector = [weight_a1, weight_a2]
        # # vector = torch.cat(vector, dim=1)
        # vector = self.softmax(vector)
        # vector = vector.unsqueeze(2).unsqueeze(3)
        #
        # for i in range(2):
        #     if i == 0:
        #         feature_map_a1 = feature_map_a1 * vector[:, i, :, :].unsqueeze(3)
        #     if i == 1:
        #         feature_map_a2 = feature_map_a2 * vector[:, i, :, :].unsqueeze(3)
        #
        # feature_map_2 = feature_map_a1 + feature_map_a2

        # feature_map_2 = (feature_map_a1 + feature_map_a2) / 2

        # 自适应两feature map比例
        # weight_f1 = self.avg(feature_map_1).squeeze(2).squeeze(2)
        # weight_f1 = self.linear3(weight_f1)
        # weight_f2 = self.avg(feature_map_2).squeeze(2).squeeze(2)
        # weight_f2 = self.linear4(weight_f2)
        #
        # vector = [weight_f1, weight_f2]
        # vector = torch.cat(vector, dim=1)
        # vector = self.softmax(vector)
        # vector = vector.unsqueeze(2).unsqueeze(3)
        #
        # for i in range(2):
        #     if i == 0:
        #         feature_map_1 = feature_map_1 * vector[:, i, :, :].unsqueeze(3)
        #     if i == 1:
        #         feature_map_2 = feature_map_2 * vector[:, i, :, :].unsqueeze(3)
        #
        # feature_map = feature_map_1 + feature_map_2 + feature_map
        feature_map = feature_map_1 + feature_map
        # feature_map = torch.cat((feature_map_1, feature_map_global), dim=1)

        return feature_map, 0


class DynamicConv4(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv4, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial3 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )
        self.spatial4 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )
        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.FA2 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((7, 7))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

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

        weight_vector = [weight1, weight2, weight3, weight4]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(4):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)

        # feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        temp1 = torch.cat((feature_map1, feature_map2), 2)
        temp2 = torch.cat((feature_map3, feature_map4), 2)
        feature_map = torch.cat((temp1, temp2), 3)
        # feature_map_global = feature_map1 + feature_map2 + feature_map3 + feature_map4
        # feature_map_global = self.upsample(feature_map_global)

        # 串联，SA+FA
        bs, c, _, _ = feature_map.shape
        feature_map_1 = feature_map.view(bs, c, -1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        feature_map_1, _ = self.SA1(feature_map_1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        feature_map_1 = feature_map_1.view(bs, c, 14, 14)

        # 加权操作，SA，FA
        # bs, c, _, _ = feature_map.shape
        # feature_map_a1 = feature_map
        # feature_map_a1 = feature_map_a1.view(bs, c, -1)
        # feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        # feature_map_a1, _ = self.SA2(feature_map_a1)
        # feature_map_a1 = feature_map_a1.permute(0, 2, 1)
        # feature_map_a1 = feature_map_a1.view(bs, c, 14, 14)
        #
        # feature_map_a2 = feature_map
        # feature_map_a2 = feature_map_a2.view(bs, c, -1)
        # feature_map_a2, score = self.FA2(feature_map_a2)
        # feature_map_a2 = feature_map_a2.view(bs, c, 14, 14)
        #
        # weight_a1 = self.avg(feature_map_a1).squeeze(2).squeeze(2)
        # weight_a1 = self.linear1(weight_a1)
        # weight_a2 = self.avg(feature_map_a2).squeeze(2).squeeze(2)
        # weight_a2 = self.linear2(weight_a2)
        #
        # vector = [weight_a1, weight_a2]
        # # vector = torch.cat(vector, dim=1)
        # vector = self.softmax(vector)
        # vector = vector.unsqueeze(2).unsqueeze(3)
        #
        # for i in range(2):
        #     if i == 0:
        #         feature_map_a1 = feature_map_a1 * vector[:, i, :, :].unsqueeze(3)
        #     if i == 1:
        #         feature_map_a2 = feature_map_a2 * vector[:, i, :, :].unsqueeze(3)
        #
        # feature_map_2 = feature_map_a1 + feature_map_a2

        # feature_map_2 = (feature_map_a1 + feature_map_a2) / 2

        # 自适应两feature map比例
        # weight_f1 = self.avg(feature_map_1).squeeze(2).squeeze(2)
        # weight_f1 = self.linear3(weight_f1)
        # weight_f2 = self.avg(feature_map_2).squeeze(2).squeeze(2)
        # weight_f2 = self.linear4(weight_f2)
        #
        # vector = [weight_f1, weight_f2]
        # vector = torch.cat(vector, dim=1)
        # vector = self.softmax(vector)
        # vector = vector.unsqueeze(2).unsqueeze(3)
        #
        # for i in range(2):
        #     if i == 0:
        #         feature_map_1 = feature_map_1 * vector[:, i, :, :].unsqueeze(3)
        #     if i == 1:
        #         feature_map_2 = feature_map_2 * vector[:, i, :, :].unsqueeze(3)
        #
        # feature_map = feature_map_1 + feature_map_2 + feature_map
        feature_map = feature_map_1 + feature_map
        # feature_map = torch.cat((feature_map_1, feature_map_global), dim=1)

        return feature_map, 0


class AConv_DHSA_global_x_11(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(AConv_DHSA_global_x_11, self).__init__()
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
        self.conv1 = DynamicConv11(args, 64, 4, 0.7)
        self.conv2 = DynamicConv11(args, 128, 4, 0.8)
        self.conv3 = DynamicConv11(args, 256, 4, 0.9)
        self.conv4 = DynamicConv11(args, 512, 4, 1)

        self.select1 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select2 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select3 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select4 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(1024, num_classes)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = self.fc.weight.size(1)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, f1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        featuremap1 = featureMap1
        importance1 = self.select1(self.avgpool(featureMap1).view(bs, -1)).unsqueeze(2)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, f2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        featuremap2 = featureMap2
        importance2 = self.select2(self.avgpool(featureMap2).view(bs, -1)).unsqueeze(2)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, f3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        featuremap3 = featureMap3
        importance3 = self.select3(self.avgpool(featureMap3).view(bs, -1)).unsqueeze(2)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, f4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        featuremap4 = featureMap4
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

        featuremap = featureMap1 + featureMap2 + featureMap3 + featureMap4

        featureMap = self.avgpool(featuremap4)
        x = self.avgpool(x)
        feature = torch.cat((featureMap, x), dim=1)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return out, featuremap, featuremap1, featuremap2, featuremap3, featuremap4, feature

    def output_num(self):
        return self.output_dim


class DynamicConv11(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv11, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial3 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial4 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )
        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.FA2 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((14, 14))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        if x.size(2) == 7:
            x = self.upsample(x)
        else:
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

        weight_vector = [weight1, weight2, weight3, weight4]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(4):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                bs, c, _, _ = feature_map1.shape
                feature_map_1 = feature_map1.view(bs, c, -1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                feature_map_1, f = self.SA1(feature_map_1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                # feature_map_1, _ = self.FA1(feature_map_1)
                feature_map1 = feature_map_1.view(bs, c, 14, 14)

            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                bs, c, _, _ = feature_map2.shape
                feature_map_1 = feature_map2.view(bs, c, -1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                feature_map_1, f = self.SA1(feature_map_1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                # feature_map_1, _ = self.FA1(feature_map_1)
                feature_map2 = feature_map_1.view(bs, c, 14, 14)

            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                bs, c, _, _ = feature_map3.shape
                feature_map_1 = feature_map3.view(bs, c, -1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                feature_map_1, f = self.SA1(feature_map_1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                # feature_map_1, _ = self.FA1(feature_map_1)
                feature_map3 = feature_map_1.view(bs, c, 14, 14)

            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
                bs, c, _, _ = feature_map4.shape
                feature_map_1 = feature_map4.view(bs, c, -1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                feature_map_1, f = self.SA1(feature_map_1)
                feature_map_1 = feature_map_1.permute(0, 2, 1)
                # feature_map_1, _ = self.FA1(feature_map_1)
                feature_map4 = feature_map_1.view(bs, c, 14, 14)

        feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        return feature_map, f


class AConv_DHSA_global_x_21(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(AConv_DHSA_global_x_21, self).__init__()
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
        self.conv1 = DynamicConv21(args, 64, 4, 0.7)
        self.conv2 = DynamicConv21(args, 128, 4, 0.8)
        self.conv3 = DynamicConv21(args, 256, 4, 0.9)
        self.conv4 = DynamicConv21(args, 512, 4, 1)

        self.select1 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select2 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select3 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select4 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(1024, num_classes)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = self.fc.weight.size(1)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, f1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        featuremap1 = featureMap1
        importance1 = self.select1(self.avgpool(featureMap1).view(bs, -1)).unsqueeze(2)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, f2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        featuremap2 = featureMap2
        importance2 = self.select2(self.avgpool(featureMap2).view(bs, -1)).unsqueeze(2)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, f3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        featuremap3 = featureMap3
        importance3 = self.select3(self.avgpool(featureMap3).view(bs, -1)).unsqueeze(2)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, f4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        featuremap4 = featureMap4
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

        featuremap = featureMap1 + featureMap2 + featureMap3 + featureMap4

        featureMap = self.avgpool(featuremap4)
        x = self.avgpool(x)
        feature = torch.cat((featureMap, x), dim=1)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return out, featuremap, featuremap1, featuremap2, featuremap3, featuremap4, feature

    def output_num(self):
        return self.output_dim


class DynamicConv21(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv21, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial3 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial4 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )
        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block(dim=in_feature, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.FA2 = Block(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((14, 14))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        if x.size(2) == 7:
            x = self.upsample(x)
        else:
            x = self.resize(x)

        # distribution dropout
        bs, c, _, _ = x.shape
        feature_map_1 = x.view(bs, c, -1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        feature_map_1, f = self.SA1(feature_map_1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        feature_map_1 = feature_map_1.view(bs, c, 14, 14)

        x = feature_map_1 + x

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

        weight_vector = [weight1, weight2, weight3, weight4]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(4):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)

        feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        return feature_map, f


class AConv_DHSA_global_x_22(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(AConv_DHSA_global_x_22, self).__init__()
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
        self.conv1 = DynamicConv22(args, 64, 4, 0.7)
        self.conv2 = DynamicConv22(args, 128, 4, 0.8)
        self.conv3 = DynamicConv22(args, 256, 4, 0.9)
        self.conv4 = DynamicConv22(args, 512, 4, 1)

        self.select1 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select2 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select3 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select4 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(1024, num_classes)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = self.fc.weight.size(1)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, f1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        featuremap1 = featureMap1
        importance1 = self.select1(self.avgpool(featureMap1).view(bs, -1)).unsqueeze(2)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, f2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        featuremap2 = featureMap2
        importance2 = self.select2(self.avgpool(featureMap2).view(bs, -1)).unsqueeze(2)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, f3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        featuremap3 = featureMap3
        importance3 = self.select3(self.avgpool(featureMap3).view(bs, -1)).unsqueeze(2)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, f4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        featuremap4 = featureMap4
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

        featuremap = featureMap1 + featureMap2 + featureMap3 + featureMap4

        featureMap = self.avgpool(featuremap4)
        x = self.avgpool(x)
        feature = torch.cat((featureMap, x), dim=1)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return out, featuremap, featuremap1, featuremap2, featuremap3, featuremap4, feature

    def output_num(self):
        return self.output_dim


class DynamicConv22(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv22, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial2 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial3 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )

        self.spatial4 = nn.Sequential(
            nn.Conv2d(512, 512 // ratio, 1, 1, 0),
            nn.BatchNorm2d(512 // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.channel4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // ratio, 512),
            nn.Sigmoid()
        )
        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight3 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight4 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block22(dim=in_feature, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.SA2 = Block22(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)
        self.FA1 = Block22(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.FA2 = Block22(dim=196, num_heads=28, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=512)
        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((14, 14))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        if x.size(2) == 7:
            x = self.upsample(x)
        else:
            x = self.resize(x)

        # distribution dropout
        bs, c, _, _ = x.shape
        feature_map_1 = x.view(bs, c, -1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        feature_map_1, f = self.SA1(feature_map_1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        feature_map_1 = feature_map_1.view(bs, c, 14, 14)

        x = feature_map_1 + x

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

        weight_vector = [weight1, weight2, weight3, weight4]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(4):
            if i == 0:
                feature_map1 = feature_map1 * (channel_att1.unsqueeze(2).unsqueeze(3) + spatial_att1) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 1:
                feature_map2 = feature_map2 * (channel_att2.unsqueeze(2).unsqueeze(3) + spatial_att2) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 2:
                feature_map3 = feature_map3 * (channel_att3.unsqueeze(2).unsqueeze(3) + spatial_att3) \
                               * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 3:
                feature_map4 = feature_map4 * (channel_att4.unsqueeze(2).unsqueeze(3) + spatial_att4) \
                               * weight_vector[:, i, :, :].unsqueeze(3)

        feature_map = feature_map1 + feature_map2 + feature_map3 + feature_map4

        return feature_map, f


class SAttention22(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 head_drop_ratio=0.,
                 num_patches=196):
        super(SAttention22, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.head_drop_ratio = head_drop_ratio
        # self.fc = nn.Sequential(
        #     nn.Linear(num_patches * num_patches, num_patches),
        #     nn.ReLU(),
        #     nn.Linear(num_patches, 1))
        self.fc = nn.Linear(num_patches * num_patches, 1)
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
        # print(attn)

        '''概率丢头'''
        # attn_ = attn.view(B, self.num_heads, -1)
        # prob = torch.zeros(1, self.num_heads) + 0.5 # 0.5概率丢头
        # prob = nn.init.normal_(torch.randn(B, 16), mean=0.5, std=1 / math.sqrt(12)) # 正态丢头
        # prob = nn.init.uniform_(torch.randn(B, 16), 0, 1)
        # prob = []
        # for i in range(B):
        #     for j in range(16):
        #         temp = nn.init.normal_(torch.randn(1), mean=0, std=float(2 / (N * N) * torch.norm(attn_[i][j])))  # 正态丢头
        #         prob.append(temp)
        # prob = torch.stack(prob).view(B, 16)
        # prob = prob.unsqueeze(2).cuda()
        # temp = torch.clamp(temp, 0, 1)    # 均匀丢头
        # prob = torch.bernoulli(temp)
        # score = self.fc(attn_).cuda()
        # mask = (prob > score).type(torch.uint8)
        # mask = mask.unsqueeze(3).cuda()
        # attn = attn * mask

        '''mask 丢头'''
        attn_ = attn.view(B, self.num_heads, -1)
        importance = torch.mean(attn_, 2, True)
        score = self.fc(attn_)
        mask = (importance > score).type(torch.uint8)
        # print(torch.sum(mask))
        mask = mask.unsqueeze(3)
        attn = attn * mask

        # mask = []
        # for i in range(B):
        #     for j in range(16):
        #         temp = nn.init.normal_(torch.randn(1), mean=0, std=float(2 / (N * N) * torch.norm(attn_[i][j])))  # 正态丢头
        #         mask.append(temp)
        # mask = torch.stack(mask)
        # mask = mask.view(B, -1).unsqueeze(2).unsqueeze(3)
        # mask = mask.cuda()
        # attn = attn * mask

        '''gumbel softmax'''

        # def sample_gumbel(shape, eps=1e-20):
        #     U = torch.rand(shape)
        #     U = U.cuda()
        #     return -torch.log(-torch.log(U + eps) + eps)
        #
        # def gumbel_softmax_sample(logits, temperature=1):
        #     y = logits + sample_gumbel(logits.size())
        #     return F.softmax(y / temperature, dim=-1)
        #
        # def gumbel_softmax(logits, temperature=1, hard=False):
        #     """
        #     ST-gumple-softmax
        #     input: [*, n_class]
        #     return: flatten --> [*, n_class] an one-hot vector
        #     """
        #     y = gumbel_softmax_sample(logits, temperature)
        #
        #     if not hard:
        #         return y
        #
        #     shape = y.size()
        #     _, ind = y.max(dim=-1)
        #     y_hard = torch.zeros_like(y).view(-1, shape[-1])
        #     y_hard.scatter_(1, ind.view(-1, 1), 1)
        #     y_hard = y_hard.view(*shape)
        #     # Set gradients w.r.t. y_hard gradients w.r.t. y
        #     y_hard = (y_hard - y).detach() + y
        #     return y_hard
        #
        # attn_ = attn.view(B, self.num_heads, -1)
        # importance = torch.mean(attn_, 2, True)
        # score = self.fc(attn_)
        # mask = (importance > score).type(torch.uint8)
        # # print(torch.sum(mask))
        # mask = gumbel_softmax(mask)
        # attn = attn * mask.unsqueeze(3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = self.attn_drop(attn) + self.bias

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, v.transpose(1, 2).reshape(B, N, C).permute(0, 2, 1).view(B, C, 14, 14)


class Block22(nn.Module):
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
        super(Block22, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SAttention22(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
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


class AConv_DHSA_global_x_SK(nn.Module):
    def __init__(self, args, NumOfLayer, pretrained=True, num_classes=7, drop_rate=0):
        super(AConv_DHSA_global_x_SK, self).__init__()
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
        self.conv1 = DynamicConv_SK(args, 64, 4, 0.7)
        self.conv2 = DynamicConv_SK(args, 128, 4, 0.8)
        self.conv3 = DynamicConv_SK(args, 256, 4, 0.9)
        self.conv4 = DynamicConv_SK(args, 512, 4, 1)

        self.select1 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select2 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select3 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )
        self.select4 = nn.Sequential(
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 4, 512),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(1024, num_classes)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = self.fc.weight.size(1)

    def forward(self, x):
        x = self.pre_conv(x)
        bs, c, h, w = x.shape[0], 768, 14, 14

        x = self.layer1(x)
        featureMap1 = x
        featureMap1 = self.avgpool1(featureMap1)
        featureMap1, f1 = self.conv1(featureMap1)
        featureMap1 = self.bn(featureMap1)
        featureMap1 = self.relu(featureMap1)
        featuremap1 = featureMap1
        importance1 = self.select1(self.avgpool(featureMap1).view(bs, -1)).unsqueeze(2)

        x = self.layer2(x)
        featureMap2 = x
        featureMap2 = self.avgpool1(featureMap2)
        featureMap2, f2 = self.conv2(featureMap2)
        featureMap2 = self.bn(featureMap2)
        featureMap2 = self.relu(featureMap2)
        featuremap2 = featureMap2
        importance2 = self.select2(self.avgpool(featureMap2).view(bs, -1)).unsqueeze(2)

        x = self.layer3(x)
        featureMap3 = x
        featureMap3 = self.avgpool1(featureMap3)
        featureMap3, f3 = self.conv3(featureMap3)
        featureMap3 = self.bn(featureMap3)
        featureMap3 = self.relu(featureMap3)
        featuremap3 = featureMap3
        importance3 = self.select3(self.avgpool(featureMap3).view(bs, -1)).unsqueeze(2)

        x = self.layer4(x)
        featureMap4 = x
        featureMap4 = self.upsample(featureMap4)
        featureMap4, f4 = self.conv4(featureMap4)
        featureMap4 = self.bn(featureMap4)
        featureMap4 = self.relu(featureMap4)
        featuremap4 = featureMap4
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

        featuremap = featureMap1 + featureMap2 + featureMap3 + featureMap4

        featureMap = self.avgpool(featuremap4)
        x = self.avgpool(x)
        feature = torch.cat((featureMap, x), dim=1)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return out, featuremap, featuremap1, featuremap2, featuremap3, featuremap4, feature

    def output_num(self):
        return self.output_dim


class DynamicConv_SK(nn.Module):
    def __init__(self, args, in_feature, ratio=4, drop_ratio=0):
        super(DynamicConv_SK, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, 512, 5, 1, 2)
        self.down = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Linear(512 // ratio, 512)
        )
        self.up2 = nn.Sequential(
            nn.Linear(512 // ratio, 512)
        )
        self.weight1 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )
        self.weight2 = nn.Sequential(
            nn.Linear(512, 512 // ratio),
            nn.BatchNorm1d(512 // ratio),
            nn.Tanh(),
            nn.Linear(512 // ratio, 1),
        )

        self.SA1 = Block(dim=512, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0,
                         attn_drop_ratio=0, drop_path_ratio=0, head_drop_ratio=drop_ratio,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, num_patches=196)

        self.softmax = nn.Softmax(dim=1)
        self.resize = nn.AdaptiveAvgPool2d((14, 14))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d((14, 14))
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        if x.size(2) == 7:
            x = self.upsample(x)
        else:
            x = self.resize(x)

        feature_map1 = self.conv1(x)
        feature_map2 = self.conv2(x)
        f1 = self.avg(feature_map1).view(bs, -1)
        f2 = self.avg(feature_map2).view(bs, -1)
        w1 = self.weight1(f1)
        w2 = self.weight2(f2)

        feature_map = feature_map1 + feature_map2
        s = self.avg(feature_map).view(bs, -1)
        z = self.down(s)
        a = self.up1(z)
        b = self.up2(z)

        weight_vector = [a, b]
        weight_vector = torch.cat(weight_vector, dim=1)
        weight_vector = self.softmax(weight_vector)
        weight_vector = weight_vector.unsqueeze(2).unsqueeze(3)

        for i in range(2):
            if i == 0:
                feature_map1 = feature_map1 * weight_vector[:, i, :, :].unsqueeze(3)
            elif i == 1:
                feature_map2 = feature_map2 * weight_vector[:, i, :, :].unsqueeze(3)

        feature_map = feature_map1 + feature_map2

        # distribution dropout
        bs, c, _, _ = feature_map.shape
        feature_map_1 = feature_map.view(bs, c, -1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        feature_map_1, f = self.SA1(feature_map_1)
        feature_map_1 = feature_map_1.permute(0, 2, 1)
        # feature_map_1, _ = self.FA1(feature_map_1)
        feature_map_1 = feature_map_1.view(bs, c, 14, 14)

        feature_map = feature_map_1 + feature_map

        return feature_map, f


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
        # self.fc = nn.Sequential(
        #     nn.Linear(num_patches * num_patches, num_patches),
        #     nn.ReLU(),
        #     nn.Linear(num_patches, 1))
        self.fc = nn.Linear(num_patches * num_patches, 1)
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
        # print(attn)

        '''概率丢头'''
        attn_ = attn.view(B, self.num_heads, -1)
        prob = torch.zeros(1, self.num_heads) + 0.5 # 0.5概率丢头
        # prob = nn.init.normal_(torch.randn(B, 16), mean=0.5, std=1 / math.sqrt(12)) # 正态丢头
        # prob = nn.init.uniform_(torch.randn(B, 16), 0, 1)
        # prob = []
        # for i in range(B):
        #     for j in range(16):
        #         temp = nn.init.normal_(torch.randn(1), mean=0, std=float(2 / (N * N) * torch.norm(attn_[i][j])))  # 正态丢头
        #         prob.append(temp)
        # prob = torch.stack(prob).view(B, 16)
        prob = prob.unsqueeze(2).cuda()
        # temp = torch.clamp(temp, 0, 1)    # 均匀丢头
        # prob = torch.bernoulli(temp)
        score = self.fc(attn_).cuda()
        mask = (prob > score).type(torch.uint8)
        mask = mask.unsqueeze(3).cuda()
        attn = attn * mask

        '''mask 丢头'''
        # attn_ = attn.view(B, self.num_heads, -1)
        # importance = torch.mean(attn_, 2, True)
        # score = self.fc(attn_)
        # mask = (importance > score).type(torch.uint8)
        # # print(torch.sum(mask))
        # mask = mask.unsqueeze(3)
        # attn = attn * mask

        # mask = []
        # for i in range(B):
        #     for j in range(16):
        #         temp = nn.init.normal_(torch.randn(1), mean=0, std=float(2 / (N * N) * torch.norm(attn_[i][j])))  # 正态丢头
        #         mask.append(temp)
        # mask = torch.stack(mask)
        # mask = mask.view(B, -1).unsqueeze(2).unsqueeze(3)
        # mask = mask.cuda()
        # attn = attn * mask

        '''gumbel softmax'''

        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape)
            U = U.cuda()
            return -torch.log(-torch.log(U + eps) + eps)

        def gumbel_softmax_sample(logits, temperature=1):
            y = logits + sample_gumbel(logits.size())
            return F.softmax(y / temperature, dim=-1)

        def gumbel_softmax(logits, temperature=1, hard=False):
            """
            ST-gumple-softmax
            input: [*, n_class]
            return: flatten --> [*, n_class] an one-hot vector
            """
            y = gumbel_softmax_sample(logits, temperature)

            if not hard:
                return y

            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            # Set gradients w.r.t. y_hard gradients w.r.t. y
            y_hard = (y_hard - y).detach() + y
            return y_hard

        attn_ = attn.view(B, self.num_heads, -1)
        importance = torch.mean(attn_, 2, True)
        score = self.fc(attn_)
        mask = (importance > score).type(torch.uint8)
        # print(torch.sum(mask))
        mask = gumbel_softmax(mask)
        attn = attn * mask.unsqueeze(3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = self.attn_drop(attn) + self.bias

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, v.transpose(1, 2).reshape(B, N, C).permute(0, 2, 1).view(B, C, 14, 14)


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
