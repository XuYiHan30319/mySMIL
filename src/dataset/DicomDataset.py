import torch.utils.data as data
import os
import torch
import numpy as np
import SimpleITK as sitk
from monai.transforms import Compose, RandRotate, RandFlip


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
        path="../../data/lung_dicom",
        mode="train",
    ):
        self.mode = mode
        self.img_list = []
        path = os.path.join(path, mode)
        for root, dirs, files in os.walk(os.path.join(path, "0")):
            for file in files:
                if file.endswith(".nii"):
                    self.img_list.append([os.path.join(root, file), 0])
        print("0的个数为:", len(self.img_list))
        l = len(self.img_list)
        for root, dirs, files in os.walk(os.path.join(path, "1")):
            for file in files:
                if file.endswith(".nii"):
                    self.img_list.append([os.path.join(root, file), 1])
        print("1的个数为:", len(self.img_list) - l)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = sitk.ReadImage(self.img_list[index][0])
        img_list = sitk.GetArrayFromImage(img)
        img_list = np.array(img_list)
        if img_list.shape[-1] == 3:
            # 如果图像是彩色的，计算最后一个维度（颜色通道）的平均值
            img_list = np.mean(img_list, axis=-1)
        img_list = normalize(img_list)
        # transforms = Compose(
        #     [
        #         RandRotate(
        #             range_x=(-15, 15),
        #             prob=0.5,
        #             keep_size=True,
        #             padding_mode="reflection",
        #         ),  # 随机旋转-15~15度
        #     ]
        # )
        img_list = np.array(img_list)[np.newaxis, ...]  # .transpose(1, 0, 2, 3)
        # if self.mode == "train":
        #     normalized_tensor = transforms(img_list).float()
        # else:
        normalized_tensor = torch.from_numpy(img_list).float()
        return normalized_tensor, self.img_list[index][1]


if __name__ == "__main__":
    data = Dataset3d()
    print(data[0][0].shape)
    print(data[0][1])
