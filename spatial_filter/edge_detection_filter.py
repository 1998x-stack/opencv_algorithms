import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple

class EdgeDetectionFilters:
    """
    A class to perform edge detection using:
    1. Laplacian Filter
    2. Sobel Filter
    3. Prewitt Filter
    Each filter is implemented from scratch.
    """

    @staticmethod
    def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply a convolution filter to the input image.

        Args:
            image (np.ndarray): Input grayscale image.
            kernel (np.ndarray): Filter kernel (2D array).

        Returns:
            np.ndarray: Filtered image.
        """
        # 获取核的大小
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        # 填充图像
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image, dtype=np.float32)

        # 遍历图像并应用卷积
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + k_h, j:j + k_w]
                filtered_image[i, j] = np.sum(region * kernel)

        # 将结果裁剪到有效范围
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    @staticmethod
    def laplacian_filter(image: np.ndarray) -> np.ndarray:
        """
        Apply Laplacian filter for edge detection.

        Args:
            image (np.ndarray): Input grayscale image.

        Returns:
            np.ndarray: Laplacian-filtered image.
        """
        # 拉普拉斯核 (3x3)
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
        return EdgeDetectionFilters.apply_filter(image, kernel)

    @staticmethod
    def sobel_filter(image: np.ndarray, axis: str = 'x') -> np.ndarray:
        """
        Apply Sobel filter for edge detection.

        Args:
            image (np.ndarray): Input grayscale image.
            axis (str): Direction of the filter ('x' or 'y').

        Returns:
            np.ndarray: Sobel-filtered image.
        """
        # 索贝尔核 (x 和 y)
        if axis == 'x':
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
        elif axis == 'y':
            kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
        else:
            raise ValueError("Axis must be 'x' or 'y'")

        return EdgeDetectionFilters.apply_filter(image, kernel)

    @staticmethod
    def prewitt_filter(image: np.ndarray, axis: str = 'x') -> np.ndarray:
        """
        Apply Prewitt filter for edge detection.

        Args:
            image (np.ndarray): Input grayscale image.
            axis (str): Direction of the filter ('x' or 'y').

        Returns:
            np.ndarray: Prewitt-filtered image.
        """
        # Prewitt 核 (x 和 y)
        if axis == 'x':
            kernel = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])
        elif axis == 'y':
            kernel = np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]])
        else:
            raise ValueError("Axis must be 'x' or 'y'")

        return EdgeDetectionFilters.apply_filter(image, kernel)

# 工业场景示例
if __name__ == "__main__":
    # 加载灰度图像
    image_path = "example.jpg"  # 替换为实际图像路径
    original_image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

    # 应用拉普拉斯滤波器
    laplacian_image = EdgeDetectionFilters.laplacian_filter(original_image)

    # 应用索贝尔滤波器 (x 和 y)
    sobel_x_image = EdgeDetectionFilters.sobel_filter(original_image, axis='x')
    sobel_y_image = EdgeDetectionFilters.sobel_filter(original_image, axis='y')

    # 应用 Prewitt 滤波器 (x 和 y)
    prewitt_x_image = EdgeDetectionFilters.prewitt_filter(original_image, axis='x')
    prewitt_y_image = EdgeDetectionFilters.prewitt_filter(original_image, axis='y')

    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(laplacian_image, cmap="gray")
    plt.title("Laplacian Filter")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(sobel_x_image, cmap="gray")
    plt.title("Sobel Filter (X)")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(sobel_y_image, cmap="gray")
    plt.title("Sobel Filter (Y)")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(prewitt_x_image, cmap="gray")
    plt.title("Prewitt Filter (X)")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(prewitt_y_image, cmap="gray")
    plt.title("Prewitt Filter (Y)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
