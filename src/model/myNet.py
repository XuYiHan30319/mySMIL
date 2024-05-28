import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# 只是一个模版
# 假设 ResNet3d 和 ResNet2d 都已经定义好或从 torchvision.models 中加载


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.encoder = models.video.r3d_18(pretrained=True)
        self.encoder.fc = nn.Identity()  # 去掉最后的全连接层

    def forward(self, x):
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # 去掉最后的全连接层

    def forward(self, x):
        return self.encoder(x)


class RNetwork(nn.Module):
    def __init__(self, input_dim, num_priors):
        super(RNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, num_priors)  # 预测高斯分布的均值
        self.fc_logvar = nn.Linear(256, num_priors)  # 预测高斯分布的对数方差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Classifier(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, num_classes, num_priors):
        super(Classifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.r_network = RNetwork(image_feature_dim, num_priors)
        self.num_priors = num_priors
        self.priors = nn.Parameter(
            torch.randn(num_priors, text_feature_dim)
        )  # 模态先验
        self.fc = nn.Linear(image_feature_dim + text_feature_dim, num_classes)

    def forward(self, image, text=None):
        image_features = self.image_encoder(image)
        if text is not None:
            text_features = self.text_encoder(text)
        else:
            mu, logvar = self.r_network(image_features)
            weights = reparameterize(mu, logvar)
            weights = F.softmax(weights, dim=-1)  # 使用 softmax 确保权重和为1
            text_features = torch.matmul(weights, self.priors)  # 重建缺失模态

        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.fc(combined_features)
        return logits


# 假设输入的维度为
image_feature_dim = 512  # 根据 ResNet3d 的输出维度
text_feature_dim = 512  # 根据 ResNet2d 的输出维度
num_classes = 10  # 分类数量
num_priors = 20  # 假设有 20 个模态先验

model = Classifier(image_feature_dim, text_feature_dim, num_classes, num_priors)

# 示例输入
image_input = torch.randn(
    8, 3, 16, 112, 112
)  # 假设 batch_size=8, 3 通道, 16 帧, 112x112 尺寸
text_input = torch.randn(8, 3, 224, 224)  # 假设 batch_size=8, 3 通道, 224x224 尺寸

# 前向传播
output = model(image_input, text_input)
print(output.shape)  # 输出形状应为 (8, num_classes)

# 如果缺少病理图像，可以仅传递 image_input
output_without_text = model(image_input)
print(output_without_text.shape)  # 输出形状应为 (8, num_classes)
