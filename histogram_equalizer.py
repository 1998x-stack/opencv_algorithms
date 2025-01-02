import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    实现直方图均衡化，用于全局对比度增强。

    Args:
        image (np.ndarray): 输入的灰度图像。

    Returns:
        np.ndarray: 直方图均衡化后的图像。
    """
    # 应用 OpenCV 的直方图均衡化方法
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# 读取输入灰度图像
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# 应用直方图均衡化
equalized_image = histogram_equalization(image)

# 显示原图和均衡化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘制直方图对比
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Original Histogram')
plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
plt.title('Equalized Histogram')
plt.show()



from typing import List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class HistogramEqualizer:
    """
    A class to perform histogram equalization on grayscale images.
    """

    @staticmethod
    def calculate_histogram(image: np.ndarray) -> List[int]:
        """
        Calculate the histogram of a grayscale image.

        Args:
            image (np.ndarray): Input grayscale image as a 2D numpy array.

        Returns:
            List[int]: Histogram (list of pixel frequencies) with 256 bins.
        """
        # 确保输入图像是二维数组
        assert image.ndim == 2, "Input image must be a 2D array"

        # 初始化直方图数组
        histogram = [0] * 256

        # 计算每个灰度值的频率
        for pixel in image.flatten():
            histogram[pixel] += 1

        return histogram

    @staticmethod
    def calculate_cdf(histogram: List[int]) -> List[float]:
        """
        Calculate the cumulative distribution function (CDF) from a histogram.

        Args:
            histogram (List[int]): Histogram of the image.

        Returns:
            List[float]: Normalized cumulative distribution function.
        """
        # 计算直方图的累计分布函数
        cdf = np.cumsum(histogram).astype(float)

        # 归一化 CDF 到 [0, 255]
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255

        return cdf_normalized.astype(int).tolist()

    @staticmethod
    def apply_equalization(image: np.ndarray, cdf: List[int]) -> np.ndarray:
        """
        Apply histogram equalization using the CDF.

        Args:
            image (np.ndarray): Input grayscale image as a 2D numpy array.
            cdf (List[int]): Normalized cumulative distribution function.

        Returns:
            np.ndarray: Equalized grayscale image.
        """
        # 映射原图像中的每个像素值到新的像素值
        equalized_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                equalized_image[i, j] = cdf[image[i, j]]

        return equalized_image

    def equalize(self, image: np.ndarray) -> np.ndarray:
        """
        Perform histogram equalization on the input grayscale image.

        Args:
            image (np.ndarray): Input grayscale image as a 2D numpy array.

        Returns:
            np.ndarray: Equalized grayscale image.
        """
        # 计算原图的直方图
        print("Calculating histogram...")
        histogram = self.calculate_histogram(image)

        # 计算累计分布函数 (CDF)
        print("Calculating CDF...")
        cdf = self.calculate_cdf(histogram)

        # 应用直方图均衡化
        print("Applying histogram equalization...")
        equalized_image = self.apply_equalization(image, cdf)

        return equalized_image

# 工业场景中的应用示例
if __name__ == "__main__":
    # 使用 PIL 加载灰度图像
    image_path = "example.jpg"  # 替换为实际图像路径
    pil_image = Image.open(image_path).convert("L")
    example_image = np.array(pil_image, dtype=np.uint8)

    print("Original Image Loaded.")

    # 实例化直方图均衡化器
    equalizer = HistogramEqualizer()

    # 执行直方图均衡化
    print("Performing Histogram Equalization...")
    equalized_image = equalizer.equalize(example_image)

    # 可视化原图和均衡化结果
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 显示原图
    axes[0].imshow(example_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 显示均衡化后的图像
    axes[1].imshow(equalized_image, cmap='gray')
    axes[1].set_title("Equalized Image")
    axes[1].axis("off")

    # 显示结果
    plt.tight_layout()
    plt.show()