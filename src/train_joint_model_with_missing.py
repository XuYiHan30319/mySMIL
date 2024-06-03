from model import *
import torch.optim as optim
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.JointModel import JointModel
from model.ResNet3d import generate_model as generate_model3d
from model.ResNet2d import generate_model as generate_model2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.DicomDataset import Dataset3d
from sklearn.metrics import accuracy_score
import numpy as np


def train(
    path="",
    model3d_path="../model/resnet503d/resnet503d_epoch_380_best.pth",
    model2d_path="../model/resnet50/resnet50_epoch_best_210.pth",
    kmeans_path="../model/kmeans.pth",
):
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter()
    center = torch.load(kmeans_path)
    center = torch.from_numpy(center).float().to("cuda:0").T
    model = JointModel(generate_model2d(50), generate_model3d(50))
    if path != "":
        model.load_state_dict(torch.load(path))
        lunshu = int(path.split("_")[-1].split(".")[0])
        print(f"load model {lunshu} success")
    else:
        lunshu = 0
    if model3d_path != "":
        model.dicom_model.load_state_dict(torch.load(model3d_path))
    if model2d_path != "":
        model.pathology_model.load_state_dict(torch.load(model2d_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    initial_learning_rate = 5e-5
    optimizer = optim.Adam(
        model.parameters(), initial_learning_rate, betas=(0.9, 0.99), weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    num_epochs = 5000
    save_path = "../model/joint_Path/"  # 这里是保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_dataset = Dataset3d("../data/lung_dicom", "test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=16)

    train_dataset = Dataset3d("../data/lung_dicom", "train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
    batchs = len(train_loader)
    for epoch in range(lunshu, num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(
            total=batchs, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ) as pbar:
            for i, (data3d, target) in enumerate(train_loader):
                data3d, target = (
                    data3d.to(device),
                    target.to(device),
                )
                optimizer.zero_grad()
                classification_output = model(data3d, mode="one", pathology_mean=center)
                loss = criterion(classification_output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 5 == 0:
                    pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        if epoch_loss < 0.1:
            # 测试模型准确率
            model.eval()
            # 准备评价指标
            all_targets = []
            all_predictions = []

            with torch.no_grad():
                for data3d, target in test_loader:
                    data3d, target = (
                        data3d.to(device),
                        target.to(device),
                    )
                    outputs = model(data3d, mode="one", pathology_mean=center)
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
            model_save_name = f"Joint_epoch_{epoch + 1}.pth"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, model_save_name))
            print(f"Model saved as {model_save_name}")
        # 将损失写入TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)

    # 关闭SummaryWriter
    writer.close()


if __name__ == "__main__":
    train(path="../model/joint_Path/Joint_epoch_70.pth", model3d_path="")
