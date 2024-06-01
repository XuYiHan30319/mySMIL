import os
import shutil
import pandas as pd
from random import random


if __name__ == "__main__":
    path = "../../data/manifest-1608669183333/Lung-PET-CT-Dx"
    target_path = "../../data/lung_dicom"
    for root, dirs, files in os.walk(path):
        if len(files) >= 20:
            if random() > 0.8:
                tag = "test"
            else:
                tag = "train"
            temp_root = root[len(path) + 1 :]
            # 获得第一个路径
            base_name = temp_root.split("/")[0]
            if "A" in base_name:
                label = 1  # A”诊断为腺癌 LUAD
            else:
                label = 0  # B”诊断为鳞癌 LSCC 0
            target = os.path.join(target_path, tag, str(label), base_name,os.path.basename(root))
            shutil.move(root, target)
