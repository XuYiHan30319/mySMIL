import os
import pandas as pd
import shutil


if __name__ == "__main__":
    csv_path = "../../data/manifest-1677266205028/metadata.csv"
    base_path = "../../data/manifest-1677266205028"
    target_path = "../../data/lung_dicom/0"
    df = pd.read_csv(csv_path)
    # 获得指定列
    df = df[["Series Description", "File Location", "Subject ID"]]
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
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # 得到series Description包括在series中的File Location
    df = df[df["Series Description"].isin(lung_series_descriptions)]
    for index, row in df.iterrows():
        file_location = row["File Location"].replace("\\", "/")
        Subject_ID = row["Subject ID"]
        source = os.path.join(base_path, file_location)
        destination = os.path.join(
            target_path, Subject_ID, os.path.basename(file_location)
        )
        shutil.move(source, destination)
