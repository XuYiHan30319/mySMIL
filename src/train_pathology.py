from model import *
import torch.optim as optim
import torch.utils.data as data
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.ResNet2d import generate_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.PathologyDataset import PathologyDataset
from sklearn.metrics import accuracy_score
import numpy as np


def train(path=""):
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter()
    model = generate_model(50, 2)
    if path != "":
        model.load_state_dict(torch.load(path))
        lunshu = int(path.split("_")[-1].split(".")[0])
        print(f"load model {lunshu} success")
    else:
        lunshu = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    initial_learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), initial_learning_rate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    num_epochs = 5000
    save_path = "./model/"  # 这里是保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_dataset = PathologyDataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    batchs = len(train_loader)
    for epoch in range(lunshu, num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(
            total=batchs, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ) as pbar:
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                classification_output = model(data)
                loss = criterion(classification_output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
                if (i + 1) % 5 == 0:
                    pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
        scheduler.step(epoch)
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        if (epoch + 1) % 10 == 0:
            model_save_name = f"resnet50_epoch_{epoch + 1}.pth"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, model_save_name))
            print(f"Model saved as {model_save_name}")

        # 将损失写入TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)

    # 关闭SummaryWriter
    writer.close()


def test():
    model = generate_model(50).to("cuda:0")
    sample = torch.rand(1, 3, 224, 224).to("cuda:0")
    res = model.forward(sample, mode="two")
    print(res.shape)


def eval(model_path="", dataset_path=""):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 载入模型
    model = generate_model(50, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 载入数据集
    eval_dataset = PathologyDataset(path=dataset_path)  # 这里传入数据集路径
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=8)

    # 准备评价指标
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算每个类的准确率
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    for class_index in range(2):
        class_targets = (all_targets == class_index).astype(int)
        class_predictions = (all_predictions == class_index).astype(int)
        accuracy = accuracy_score(class_targets, class_predictions)
        print(f"Class {class_index} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train()
    # test()
    # eval()
