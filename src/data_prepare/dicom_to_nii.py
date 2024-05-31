import SimpleITK as sitk
import os


def resampleToFixedSize(sitkImage, target_size=(224, 224, 100)):
    """
    将图像缩放到固定尺寸
    """
    euler3d = sitk.Euler3DTransform()

    # 获取原始尺寸、间隔、原点和方向
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()

    # 计算新的间隔
    new_spacing = (
        xspacing * xsize / target_size[0],
        yspacing * ysize / target_size[1],
        zspacing * zsize / target_size[2],
    )

    # 使用线性插值对图像进行重采样
    sitkImage = sitk.Resample(
        sitkImage,
        target_size,
        euler3d,
        sitk.sitkLinear,
        origin,
        new_spacing,
        direction,
    )

    return sitkImage


if __name__ == "__main__":
    path = "../../data/lung_dicom"
    # 把dicom文件转换为nii文件
    for root, dirs, files in os.walk(path):
        if files.__len__() == 0:
            continue
        # print(root)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(root)
        # 打印尺寸
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        print(image.GetSize())
        image = resampleToFixedSize(image)
        sitk.WriteImage(image, root + ".nii")
