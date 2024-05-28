import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)
from model.ResNet2d import resnet34
from clean_dataset import CleanDataset
from torch.utils.data import DataLoader
import torch


def train():
    dataset = CleanDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=2, include_top=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
