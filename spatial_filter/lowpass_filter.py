import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple

class LowPassFilter:
    """
    A class to apply different types of low-pass filters for image smoothing.
    Includes:
    - Mean Filter
    - Median Filter
    - Gaussian Filter
    """

    @staticmethod
    def mean_filter(image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply mean (average) filter to the image.

        Args:
            image (np.ndarray): Input grayscale image.
            kernel_size (Tuple[int, int]): Size of the filtering kernel.

        Returns:
            np.ndarray: Filtered image.
        """
        # 获取核大小
        k_h, k_w = kernel_size
        padded_image = np.pad(image, ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image, dtype=np.float32)

        # 遍历图像
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + k_h, j:j + k_w]
                filtered_image[i, j] = np.mean(region)

        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply median filter to the image.

        Args:
            image (np.ndarray): Input grayscale image.
            kernel_size (int): Size of the filtering kernel (assumed square).

        Returns:
            np.ndarray: Filtered image.
        """
        k = kernel_size
        padded_image = np.pad(image, ((k // 2, k // 2), (k // 2, k // 2)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image, dtype=np.uint8)

        # 遍历图像
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + k, j:j + k]
                filtered_image[i, j] = np.median(region)

        return filtered_image

    @staticmethod
    def gaussian_filter(image: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Apply Gaussian filter to the image.

        Args:
            image (np.ndarray): Input grayscale image.
            kernel_size (Tuple[int, int]): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: Filtered image.
        """
        k_h, k_w = kernel_size
        y, x = np.mgrid[-k_h // 2 + 1:k_h // 2 + 1, -k_w // 2 + 1:k_w // 2 + 1]
        gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()

        padded_image = np.pad(image, ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image, dtype=np.float32)

        # 遍历图像
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + k_h, j:j + k_w]
                filtered_image[i, j] = np.sum(region * gaussian_kernel)

        return np.clip(filtered_image, 0, 255).astype(np.uint8)

# 工业场景示例
if __name__ == "__main__":
    # 加载灰度图像
    image_path = "images/CIFAR-10-images/train/automobile/0002.jpg"  # 替换为实际图像路径
    pil_image = Image.open(image_path).convert("L")
    input_image = np.array(pil_image, dtype=np.uint8)

    # 显示原始图像
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # 应用均值滤波器
    mean_filtered = LowPassFilter.mean_filter(input_image, kernel_size=(5, 5))
    plt.subplot(2, 2, 2)
    plt.imshow(mean_filtered, cmap="gray")
    plt.title("Mean Filter")
    plt.axis("off")

    # 应用中值滤波器
    median_filtered = LowPassFilter.median_filter(input_image, kernel_size=5)
    plt.subplot(2, 2, 3)
    plt.imshow(median_filtered, cmap="gray")
    plt.title("Median Filter")
    plt.axis("off")

    # 应用高斯滤波器
    gaussian_filtered = LowPassFilter.gaussian_filter(input_image, kernel_size=(5, 5), sigma=1.0)
    plt.subplot(2, 2, 4)
    plt.imshow(gaussian_filtered, cmap="gray")
    plt.title("Gaussian Filter")
    plt.axis("off")

    # 显示结果
    plt.tight_layout()
    plt.show()
