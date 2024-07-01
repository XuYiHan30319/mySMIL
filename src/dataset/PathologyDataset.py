import torch.utils.data as data
import albumentations as A
import os
import torch
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


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


class PathologyDataset(data.Dataset):
    def __init__(self, path="../data/pathology_img_data/", mode="train", p=0.5):
        self.mode = mode
        self.p = p
        self.img_list = []
        path = os.path.join(path, mode)
        l = 0
        for i in range(2):
            tpath = os.path.join(path, str(i))
            for folder_name, _, files in os.walk(tpath):
                for file in files:
                    subfolder_path = os.path.join(folder_name, file)
                    self.img_list.append((subfolder_path, i))
            print(f"{i}的个数为:{len(self.img_list)-l}")
            l = len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        array = np.array(Image.open(self.img_list[index][0]))
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
        return normalized_tensor, self.img_list[index][1]


if __name__ == "__main__":
    data = PathologyDataset("../../data/pathology_img_data","test")
    print(data.__len__())
