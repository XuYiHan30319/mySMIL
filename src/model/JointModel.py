import os
import torch
import numpy as np
from ResNet2d import generate_model as generate_model2d
from torch.utils.data import DataLoader
from ResNet3d import generate_model as generate_model3d
import torch.nn


class JointModel(torch.nn.Module):
    def __init__(self, pathology_model, dicom_model, extractor_grad=False):
        super(JointModel, self).__init__()
        self.pathology_model = pathology_model
        self.dicom_model = dicom_model
        self.fc1 = torch.nn.Linear(2048 * 2, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        self.dropout = torch.nn.Dropout()

        self.reconstruction = torch.nn.Sequential(
            torch.nn.Linear(2045, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )
        if not extractor_grad:  # 不训练病理特征提取器

            for p in self.pathology_model.parameters():
                p.requires_grad_(False)

    def forward(self, x3d, x2d=None, pathology_mean=None, mode="two"):
        if mode == "two":  # 全模态的训练
            x2d = self.pathology_model(x2d)
            x3d = self.dicom_model(x3d)
            x = torch.cat([x2d, x3d], dim=1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        elif mode == "one":  # 只有3d的训练
            assert pathology_mean is not None
            assert x2d is None
            x3d = self.dicom_model(x3d)

            # pathology_mean的形状为2048,2,扩展一个维度，用于向量乘法
            pathology_mean = pathology_mean.expand(
                x3d.shape[0], -1, -1
            )  # batchsize,2048,2
            weight = self.reconstruction(x3d).unsqueeze(-1)  # batch_size,2,1
            pathology_feature = pathology_mean @ weight  # bathc_size,2048,1
            for i in range(pathology_feature.shape[0]):
                pathology_feature[i] = pathology_feature[i].clone() / weight.sum(1)[i]
            pathology_feature = pathology_feature.view(pathology_feature.shape[0], -1)

            # 最后的分类
            x = torch.concat([x3d, pathology_feature], dim=1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
