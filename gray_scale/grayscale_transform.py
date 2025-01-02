import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple

class GrayscaleTransform:
    """
    A class to apply various grayscale transformations to images, including:
    - Linear transformation (brightness and contrast adjustment)
    - Logarithmic transformation (enhancing low-intensity regions)
    - Power-law (gamma) transformation (adjusting dynamic range and contrast)
    - Piecewise linear transformation (e.g., histogram equalization)
    """

    @staticmethod
    def linear_transform(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Apply a linear transformation to adjust brightness and contrast.

        Args:
            image (np.ndarray): Input grayscale image.
            alpha (float): Contrast scaling factor.
            beta (float): Brightness adjustment factor.

        Returns:
            np.ndarray: Transformed image.
        """
        # 调整亮度和对比度的线性变换: new_pixel = alpha * old_pixel + beta
        transformed_image = np.clip(alpha * image + beta, 0, 255)
        return transformed_image.astype(np.uint8)

    @staticmethod
    def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Apply a logarithmic transformation to enhance low-intensity regions.

        Args:
            image (np.ndarray): Input grayscale image.
            c (float): Scaling constant (default is 1.0).

        Returns:
            np.ndarray: Transformed image.
        """
        # 对数变换: new_pixel = c * log(1 + old_pixel)
        transformed_image = c * np.log1p(image)
        transformed_image = np.clip(transformed_image * (255 / np.log1p(255)), 0, 255)
        return transformed_image.astype(np.uint8)

    @staticmethod
    def power_law_transform(image: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
        """
        Apply a power-law (gamma) transformation.

        Args:
            image (np.ndarray): Input grayscale image.
            gamma (float): Gamma correction value.
            c (float): Scaling constant (default is 1.0).

        Returns:
            np.ndarray: Transformed image.
        """
        # 幂次变换: new_pixel = c * (old_pixel ^ gamma)
        normalized_image = image / 255.0
        transformed_image = c * (normalized_image ** gamma) * 255
        return np.clip(transformed_image, 0, 255).astype(np.uint8)

    @staticmethod
    def piecewise_linear_transform(image: np.ndarray, r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
        """
        Apply piecewise linear transformation for contrast stretching.

        Args:
            image (np.ndarray): Input grayscale image.
            r1 (int): Input intensity threshold 1.
            s1 (int): Output intensity for threshold 1.
            r2 (int): Input intensity threshold 2.
            s2 (int): Output intensity for threshold 2.

        Returns:
            np.ndarray: Transformed image.
        """
        # 分段线性变换: 根据 (r1, s1) 和 (r2, s2) 构建线性插值关系
        transformed_image = np.zeros_like(image, dtype=np.float32)

        # 第一段: [0, r1]
        mask1 = image <= r1
        transformed_image[mask1] = (s1 / r1) * image[mask1]

        # 第二段: [r1, r2]
        mask2 = (image > r1) & (image <= r2)
        transformed_image[mask2] = ((s2 - s1) / (r2 - r1)) * (image[mask2] - r1) + s1

        # 第三段: [r2, 255]
        mask3 = image > r2
        transformed_image[mask3] = ((255 - s2) / (255 - r2)) * (image[mask3] - r2) + s2

        return np.clip(transformed_image, 0, 255).astype(np.uint8)

# 工业应用示例
if __name__ == "__main__":
    # 加载灰度图像
    image_path = "images/CIFAR-10-images/train/automobile/0001.jpg"  # 替换为实际图像路径
    pil_image = Image.open(image_path).convert("L")
    input_image = np.array(pil_image, dtype=np.uint8)

    # 显示原始图像
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # 应用线性变换
    linear_image = GrayscaleTransform.linear_transform(input_image, alpha=1.2, beta=30)
    plt.subplot(3, 2, 2)
    plt.imshow(linear_image, cmap="gray")
    plt.title("Linear Transformation")
    plt.axis("off")

    # 应用对数变换
    log_image = GrayscaleTransform.log_transform(input_image, c=1.0)
    plt.subplot(3, 2, 3)
    plt.imshow(log_image, cmap="gray")
    plt.title("Log Transformation")
    plt.axis("off")

    # 应用幂次变换 (伽马校正)
    gamma_image = GrayscaleTransform.power_law_transform(input_image, gamma=2.2, c=1.0)
    plt.subplot(3, 2, 4)
    plt.imshow(gamma_image, cmap="gray")
    plt.title("Gamma Correction")
    plt.axis("off")

    # 应用分段线性变换
    piecewise_image = GrayscaleTransform.piecewise_linear_transform(input_image, r1=50, s1=30, r2=200, s2=220)
    plt.subplot(3, 2, 5)
    plt.imshow(piecewise_image, cmap="gray")
    plt.title("Piecewise Linear Transformation")
    plt.axis("off")

    # 显示结果
    plt.tight_layout()
    plt.show()