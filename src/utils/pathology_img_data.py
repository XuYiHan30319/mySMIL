import os
import random


def move_files(path, target_path, label):
    # 选择 20% 的文件夹作为测试集
    # 获得path下的所有文件夹
    dirs = os.listdir(path)
    for dir in dirs:
        if random.random() < 0.2:
            tag = "test"
        else:
            tag = "train"
        source = os.path.join(path, dir, "train", dir)
        # 移动文件夹
        target = os.path.join(target_path, tag, label)
        os.makedirs(target, exist_ok=True)
        os.system("mv " + source + " " + os.path.join(target_path, tag, label))


if __name__ == "__main__":
    move_files(
        "../../data/PKG_CPTAC_LSCC_v10/LSCC_processed",
        "../../data/pathology_img_data",
        "0",
    )
    move_files(
        "../../data/PKG_CPTAC_LUAD_v12/LUAD_processed",
        "../../data/pathology_img_data",
        "1",
    )
