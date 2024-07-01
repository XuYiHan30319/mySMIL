import os
import torch
import numpy as np
import torch.nn


class JointModel(torch.nn.Module):
    def __init__(
        self,
        pathology_model,
        dicom_model,
        pathology_model_extractor_grad=False,
        dicom_model_extractor_grad=False,
    ):
        super(JointModel, self).__init__()
        self.pathology_model = pathology_model
        self.dicom_model = dicom_model
        self.fc1 = torch.nn.Linear(2048 * 2, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()

        self.reconstruction = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 20),
        )

        if not pathology_model_extractor_grad:  # 不训练病理特征提取器
            for p in self.pathology_model.parameters():
                p.requires_grad_(False)

        if not dicom_model_extractor_grad:
            for p in self.dicom_model.parameters():
                p.requires_grad_(False)

    def forward(self, x3d, x2d=None, pathology_mean=None, mode="two"):
        if mode == "two":  # 全模态的训练
            x2d = self.pathology_model(x2d, mode="two")
            x3d = self.dicom_model(x3d, mode="two")
            x = torch.cat([x2d, x3d], dim=1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        elif mode == "one":  # 只有3d的训练
            assert pathology_mean is not None
            assert x2d is None
            x3d = self.dicom_model(x3d, mode="two")

            # pathology_mean的形状为2,2048,扩展一个维度，用于向量乘法
            pathology_mean = pathology_mean.expand(
                x3d.shape[0], -1, -1
            )  # batchsize,2048,2
            weight = self.reconstruction(x3d).unsqueeze(-1) 
            pathology_feature = pathology_mean @ weight  
            for i in range(pathology_feature.shape[0]):
                pathology_feature[i] = pathology_feature[i].clone() / weight.sum(1)[i]
            pathology_feature = pathology_feature.view(pathology_feature.shape[0], -1)
            # 最后的分类
            x = torch.concat([x3d, pathology_feature], dim=1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
