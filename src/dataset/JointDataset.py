import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import SimpleITK as sitk
from monai.transforms import Compose, RandRotate, RandFlip
import matplotlib.pyplot as plt
import torch
import albumentations as A
from PIL import Image


def normalize(img):
    max_val = np.max(img)
    min_val = np.min(img)
    img_1 = (img - min_val) / (max_val - min_val)
    return img_1


class JointDataset(Dataset):
    def __init__(
        self,
        dicom_path="../../data/lung_dicom",
        pathology_path="../../data/pathology_img_data",
        csv_path="../../data/tcia-luad-lusc-cohort.csv",
        mode="train",
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[["HasRadiology", "Case_ID", "Tumor", "Pathology"]]
        # 筛选出HasRadiology为true的数据
        self.df = self.df[self.df["HasRadiology"] == "Yes"]
        # 去lung_dicom中找到对应的文件夹
        dicom_path = os.path.join(dicom_path, mode)
        self.list = []
        # 找出所有Tumor为LUAD的数据
        zero_num = 0
        one_num = 0
        for i in range(len(self.df)):
            if self.df.iloc[i, 2] == "LUAD":
                tag = "1"
            else:
                tag = "0"
            # 从dicom_path中找有没有这个文件
            if os.path.exists(os.path.join(dicom_path, tag, self.df.iloc[i, 1])):
                # 找到该路径下所有的nii文件
                niis = [
                    f
                    for f in os.listdir(
                        os.path.join(dicom_path, tag, self.df.iloc[i, 1])
                    )
                    if f.endswith(".nii")
                ]
                if niis.__len__() == 0:
                    continue
                # 然后找到这个nii所对应的pathology文件夹
                if os.path.exists(
                    os.path.join(pathology_path, "train", tag, self.df.iloc[i, 3])
                ):
                    self.list.append(
                        (
                            os.path.join(dicom_path, tag, self.df.iloc[i, 1], niis[0]),
                            os.path.join(
                                pathology_path, "train", tag, self.df.iloc[i, 3]
                            ),
                            int(tag),
                        )
                    )
                elif os.path.exists(
                    os.path.join(pathology_path, "test", tag, self.df.iloc[i, 3])
                ):
                    self.list.append(
                        (
                            os.path.join(dicom_path, tag, self.df.iloc[i, 1], niis[0]),
                            os.path.join(
                                pathology_path, "test", tag, self.df.iloc[i, 3]
                            ),
                            int(tag),
                        )
                    )
                if tag == "0":
                    zero_num += 1
                else:
                    one_num += 1
        print(f"0的个数为:{zero_num}")
        print(f"1的个数为:{one_num}")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        dicom_path, pathology_path, tag = self.list[index]
        img = sitk.ReadImage(dicom_path)
        img_list = sitk.GetArrayFromImage(img)
        img_list = np.array(img_list)
        if img_list.shape[-1] == 3:
            # 如果图像是彩色的，计算最后一个维度（颜色通道）的平均值
            img_list = np.mean(img_list, axis=-1)
        img_list = normalize(img_list)
        if self.mode == "train":
            transforms = Compose(
                [
                    RandRotate(
                        range_x=(-15, 15),
                        prob=0.5,
                        keep_size=True,
                        padding_mode="reflection",
                    ),  # 随机旋转-15~15度
                ]
            )
            img_list = transforms(img_list).float()
        img_list = np.array(img_list)[np.newaxis, ...]  # .transpose(1, 0, 2, 3)
        img_list = torch.from_numpy(img_list).float()

        # 从pathology_path中找到随机一张图片
        pathology_files = os.listdir(pathology_path)
        pathology_file = np.random.choice(pathology_files)
        array = np.array(Image.open(pathology_file))
        array = normalize(array)
        if self.mode == "train":
            transforms = A.Compose(
                [
                    A.Rotate(limit=15, p=self.p),
                    A.HorizontalFlip(p=self.p),
                    A.VerticalFlip(p=self.p),
                ]
            )
            transformed = np.array(transforms(image=array)["image"])
            # 保存下来看看
        else:
            transformed = array
        normalized_tensor = torch.tensor(transformed, dtype=torch.float).permute(
            2, 0, 1
        )

        return img_list, normalized_tensor, tag


if __name__ == "__main__":
    dataset = JointDataset()
    print(dataset[0])
