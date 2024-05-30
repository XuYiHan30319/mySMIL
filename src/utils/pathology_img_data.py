import os
import random
import pandas as pd
import math


def move_files(path, target_path, label):
    # 选择 20% 的文件夹作为测试集
    # 获得path下的所有文件夹
    csv_path = "../../data/tcia-luad-lusc-cohort.csv"
    # 获得指定列
    df = pd.read_csv(
        csv_path,
        usecols=[
            "Slide_ID",
            "Normal_Free_of_Tumor",
            "Tumor",
            "Tumor_Percent_Tumor_Nuclei",
        ],
    )
    dirs = os.listdir(path)
    for dir in dirs:
        basename = os.path.basename(dir)
        Normal_Free_of_Tumor = df[df["Slide_ID"] == basename][
            "Normal_Free_of_Tumor"
        ].values[0]
        if not pd.isna(Normal_Free_of_Tumor):
            continue
        if random.random() < 0.2:
            tag = "test"
        else:
            tag = "train"
        source = os.path.join(path, dir)
        # 移动文件夹
        target = os.path.join(target_path, tag, label)
        os.makedirs(target, exist_ok=True)
        os.system("mv " + source + " " + target)


if __name__ == "__main__":
    move_files(
        "../../data/PKG_CPTAC_LSCC_v10/LSCC_processed/train",
        "../../data/pathology_img_data",
        "0",
    )
    move_files(
        "../../data/PKG_CPTAC_LUAD_v12/LUAD_processed/train",
        "../../data/pathology_img_data",
        "1",
    )
