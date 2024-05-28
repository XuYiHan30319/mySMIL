import cv2
import numpy as np
import os
from skimage.feature import greycomatrix, greycoprops


def calculate_texture_features(image):
    """
    计算图像的纹理特征。
    :param image: 输入图像（灰度图）。
    :return: 纹理特征的平均值。
    """
    glcm = greycomatrix(
        image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = greycoprops(glcm, "contrast")
    dissimilarity = greycoprops(glcm, "dissimilarity")
    homogeneity = greycoprops(glcm, "homogeneity")
    energy = greycoprops(glcm, "energy")
    correlation = greycoprops(glcm, "correlation")

    features = [
        contrast.mean(),
        dissimilarity.mean(),
        homogeneity.mean(),
        energy.mean(),
        correlation.mean(),
    ]
    return features


def is_blank_image(image, color_threshold=0.7, texture_threshold=0.7):
    """
    判断图片是否为空白或脏数据。
    :param image: 输入图像（BGR格式）。
    :param color_threshold: 颜色阈值，用于判断图像的非白色像素比例。
    :param texture_threshold: 纹理阈值，用于判断图像的纹理特征。
    :return: 如果是空白图像，返回True；否则返回False。
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算颜色特征
    non_white_pixels = np.sum(gray_image < 250)
    total_pixels = gray_image.size
    non_white_ratio = non_white_pixels / total_pixels

    # 计算纹理特征
    texture_features = calculate_texture_features(gray_image)
    texture_mean = np.mean(texture_features)

    return non_white_ratio < color_threshold and texture_mean < texture_threshold


def classify_images(input_folder, output_folder):
    """
    分类病理切片图片和空白图片。
    :param input_folder: 输入图像文件夹路径。
    :param output_folder: 输出分类结果文件夹路径。包括“pathological”和“blank”两个子文件夹。
    """
    pathological_folder = os.path.join(output_folder, "pathological")
    blank_folder = os.path.join(output_folder, "blank")

    if not os.path.exists(pathological_folder):
        os.makedirs(pathological_folder)
    if not os.path.exists(blank_folder):
        os.makedirs(blank_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)
            if image is not None:
                if is_blank_image(image):
                    cv2.imwrite(os.path.join(blank_folder, filename), image)
                else:
                    cv2.imwrite(os.path.join(pathological_folder, filename), image)


if __name__ == "__main__":
    input_folder = "../../data/PKG - UPENN-GBM_v2/NDPI_images_preprocessed/7316UP-35"  # 输入图像文件夹路径
    output_folder = "../../data/temp"  # 输出分类结果文件夹路径
    classify_images(input_folder, output_folder)
