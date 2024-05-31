import os
import shutil
import pandas as pd


def undo_move_files(csv_path, base_path, target_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 获得指定列
    df = df[["Series Description", "File Location", "Subject ID"]]

    # 定义肺部系列描述列表
    lung_series_descriptions = [
        "LUNG 1.25mm",
        "LUNG 3.0 B70f",
        "Lung 1.0",
        "Lung 3.0",
        "REKONSTRUKCJA PLUCNA",
        "0.625 LUNG",
        "plucne 5.0 B50f",
        "plucneC 5.0 B50f",
        "Lung Recons",
        "LUNG iDose 3",
        "LUNG MIP iDose 3",
        "ARTERIAL Lung",
        "HiRes lung",
        "1.25 MM PLUCA",
        "0.625 MM LUNG",
    ]

    # 筛选包含在肺部系列描述列表中的行
    df = df[df["Series Description"].isin(lung_series_descriptions)]

    # 移动文件从目标路径回到源路径
    for index, row in df.iterrows():
        file_location = row["File Location"].replace("\\", "/")
        Subject_ID = row["Subject ID"]
        source = os.path.join(target_path, os.path.basename(file_location))
        destination = os.path.join(base_path, Subject_ID, file_location)
        shutil.move(source, destination)


if __name__ == "__main__":
    csv_path = "../../data/manifest-1677266205028/metadata.csv"
    base_path = "../../data/manifest-1677266205028"
    target_path = "../../data/lung_dicom/0"

    # 撤回移动文件的操作
    undo_move_files(csv_path, base_path, target_path)
