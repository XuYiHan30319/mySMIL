import torch.utils.data as data
import albumentations as A
import os
import torch
import numpy as np
import random
from model import *
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import SimpleITK as sitk
from monai.transforms import Compose, RandRotate, RandFlip


def downsample_matrix(matrix):
    # 使用zoom函数，只在第二和第三维度上缩放
    return zoom(matrix, (1, 0.5, 0.5))


def normalize_zscore(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-4)


def normalize(img):
    max_val = np.max(img)
    min_val = np.min(img)
    img_1 = (img - min_val) / (max_val - min_val)
    return img_1


class Dataset3d(data.Dataset):
    def __init__(
        self,
        path="./Data/3d",
        mode="train",
        strengh=True,
        rand=False,
    ):
        self.strength = strengh
        self.mode = mode
        self.img_list = []
        zero_num = 0
        for prespective in ["cor"]:
            tpath = os.path.join(path, mode, prespective)
            for folder_name, subfolders, files in os.walk(tpath + "/0"):
                for file in files:
                    subfolder_path = os.path.join(folder_name, file)
                    self.img_list.append((subfolder_path, 0))
        zero_num += len(self.img_list)
        print(f"0的个数为:{len(self.img_list)}")
        for prespective in ["cor"]:
            tpath = os.path.join(path, mode, prespective)
            for i in range(1, 4):
                for folder_name, subfolders, files in os.walk(tpath + "/" + str(i)):
                    for file in files:
                        if rand:
                            if random.random() < 0.4:
                                subfolder_path = os.path.join(folder_name, file)
                                self.img_list.append((subfolder_path, 1))
                        else:
                            subfolder_path = os.path.join(folder_name, file)
                            self.img_list.append((subfolder_path, 1))
        print(f"1-3的个数为:{len(self.img_list) - zero_num}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = sitk.ReadImage(self.img_list[index][0])
        img_list = sitk.GetArrayFromImage(img)
        img_list = np.array(img_list)
        # img_list = downsample_matrix(img_list) #这个在数据准备部分那里直接做了
        # 取中心的96张图片
        img_list = img_list[img_list.shape[0] // 2 - 48 : img_list.shape[0] // 2 + 48]
        img_list = normalize(img_list)
        if self.img_list[index][1] == 0:
            p = 0.6
        else:
            p = 0.2

        transforms = Compose(
            [
                RandRotate(
                    range_x=(-15, 15),
                    prob=p,
                    keep_size=True,
                    padding_mode="reflection",
                ),  # 随机旋转-15~15度
                # RandRotate(
                #     range_y=(-5, 5),
                #     prob=p,
                #     keep_size=True,
                #     padding_mode="reflection",
                # ),
                # RandRotate(
                #     range_z=(-5, 5),
                #     prob=p,
                #     keep_size=True,
                #     padding_mode="reflection",
                # ),
                RandFlip(prob=p, spatial_axis=0),  # 在z轴上随机翻转，数据是c*d*h*w
                RandFlip(prob=p, spatial_axis=1),  # 在x轴上随机翻转
                RandFlip(prob=p, spatial_axis=2),  # 在y轴上随机翻转
            ]
        )
        img_list = np.array(img_list)[np.newaxis, ...]  # .transpose(1, 0, 2, 3)
        if self.mode == "train" and self.strength:
            if self.img_list[index][1] == 0:
                normalized_tensor = transforms(img_list).float()
            else:
                normalized_tensor = transforms(img_list).float()
            # 将数据保存下来看看
            img = normalized_tensor[0]
            for i in range(96):
                os.makedirs(
                    f"./temp/{self.mode}/{self.img_list[index][1]}", exist_ok=True
                )
                plt.imsave(
                    f"./temp/{self.mode}/{self.img_list[index][1]}/{i}.png",
                    img[i],
                    cmap="gray",
                )
        else:
            normalized_tensor = torch.from_numpy(img_list).float()
        return normalized_tensor, self.img_list[index][1]


if __name__ == "__main__":
    data = Dataset3d()
    print(data[0])
