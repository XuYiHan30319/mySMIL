from model.ResNet2d import generate_model
from dataset.PathologyDataset import PathologyDataset
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans


def get_feature_kmeans(
    data_path="../data/pathology_img_data",
    model_path="../model/resnet50/resnet50_epoch_best_210.pth",
    save_path="../model/kmeans.pth",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = generate_model(50).to(device)
    # 载入参数
    model.load_state_dict(torch.load(model_path))
    dataset = PathologyDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)
    model.eval()
    feature_list = []
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        features = model(data, mode="two")
        for feature in features:
            feature_list.append(feature.cpu().detach().numpy())
    # 聚类
    kmeans = KMeans(n_clusters=10, random_state=0).fit(feature_list)
    # 通过model预测中心点的类别
    center = model(
        torch.from_numpy(kmeans.cluster_centers_).float().to(device), mode="three"
    )
    center = torch.argmax(center, dim=1)
    print(center)

    center = torch.tensor(center)

    # 保存中心点
    torch.save(kmeans.cluster_centers_, save_path)


def read_kmeans(save_path="../model/kmeans.pth"):
    return torch.load(save_path)


if __name__ == "__main__":
    get_feature_kmeans()
    # centers = read_kmeans()
    # print(centers)
