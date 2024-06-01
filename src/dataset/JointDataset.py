import os
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".nii"):
                    self.data.append(os.path.join(root, file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dataset = JointDataset()
    print(dataset[0])
