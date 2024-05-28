import os
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms


class CleanDataset(Dataset):
    def __init__(self, transform=transforms.ToTensor()):
        self.transform = transform
        path = "../../../data/clean_dataset"
        self.imgs = []
        for root, dirs, files in os.walk(path + "/0"):
            for file in files:
                self.imgs.append([os.path.join(root, file), 0])

        for root, dirs, files in os.walk(path + "/1"):
            for file in files:
                self.imgs.append([os.path.join(root, file), 1])

    def __getitem__(self, idx):
        img_path = self.imgs[idx][0]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.imgs[idx][1]

    def __len__(self):
        return len(self.imgs)
