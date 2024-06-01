from model import *
import torch.optim as optim
import torch.utils.data as data
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.ResNet3d import generate_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.DicomDataset import Dataset3d
from sklearn.metrics import accuracy_score
import numpy as np
from torchvision.ops import sigmoid_focal_loss


def train(path=""):
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter()
    model = generate_model(50)
    if path != "":
        model.load_state_dict(torch.load(path))
        lunshu = int(path.split("_")[-1].split(".")[0])
        print(f"load model {lunshu} success")
    else:
        lunshu = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion2 = sigmoid_focal_loss
    initial_learning_rate = 3e-4
    optimizer = optim.Adam(
        model.parameters(), initial_learning_rate, betas=(0.9, 0.99), weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    num_epochs = 5000
    save_path = "../model/resnet503d/"  # 这里是保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_dataset = Dataset3d(path="../data/lung_dicom")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)

    test_dataset = Dataset3d("../data/lung_dicom", "test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=16)
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
                loss2 = criterion(
                    classification_output, target, alpha=0.8, gamma=2, reduction="mean"
                )
                loss = loss + loss2
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        if epoch_loss < 0.2:
            # 测试模型准确率
            model.eval()
            # 准备评价指标
            all_targets = []
            all_predictions = []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)

                    all_targets.extend(target.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            # 计算每个类的准确率
            all_targets = np.array(all_targets)
            all_predictions = np.array(all_predictions)

            # 计算整体准确率
            overall_accuracy = accuracy_score(all_targets, all_predictions)
            print(f"Overall Accuracy: {overall_accuracy:.4f}")

            # 计算每个类的准确率
            for class_index in range(2):
                class_mask = all_targets == class_index
                class_targets = all_targets[class_mask]
                class_predictions = all_predictions[class_mask]
                accuracy = accuracy_score(class_targets, class_predictions)
                print(f"Class {class_index} Accuracy: {accuracy:.4f}")

        if (epoch + 1) % 10 == 0:
            model_save_name = f"resnet503d_epoch_{epoch + 1}.pth"
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
    sample = torch.rand(1, 3, 96, 224, 224).to("cuda:0")
    res = model.forward(sample, mode="two")
    print(res.shape)


def eval(model_path=""):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 载入模型
    model = generate_model(50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 载入数据集
    eval_dataset = Dataset3d(
        path="../data/lung_dicom", mode="test"
    )  # 这里传入数据集路径
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=16)

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
    print(model_path)

    # 计算整体准确率
    overall_accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # 计算每个类的准确率
    for class_index in range(2):
        class_mask = all_targets == class_index
        class_targets = all_targets[class_mask]
        class_predictions = all_predictions[class_mask]
        accuracy = accuracy_score(class_targets, class_predictions)
        print(f"Class {class_index} Accuracy: {accuracy:.4f}")


def eval_folder(path=""):
    model_list = os.listdir(path)
    model_list.sort()
    for model in model_list:
        eval(model_path=os.path.join(path, model))


if __name__ == "__main__":
    train()
    # test()
    # eval("../model/resnet503d/resnet503d_epoch_100.pth")
    # eval_folder("../model/resnet503d")
