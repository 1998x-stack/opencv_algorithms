import numpy as np
from typing import Tuple

class NonMaximumSuppression:
    """
    A class to perform Non-Maximum Suppression (NMS) for edge thinning in gradient magnitude images.
    This implementation assumes that gradient directions are provided alongside magnitudes.
    """

    @staticmethod
    def quantize_gradient_direction(direction: np.ndarray) -> np.ndarray:
        """
        Quantize gradient directions into 4 main categories: 0°, 45°, 90°, and 135°.

        Args:
            direction (np.ndarray): Gradient direction in radians.

        Returns:
            np.ndarray: Quantized gradient direction in degrees (0, 45, 90, or 135).
        """
        # 将方向从弧度转换为角度，并归一化到 [0, 180) 范围
        direction = (np.degrees(direction) + 180) % 180

        # 初始化量化方向数组为零
        quantized_direction = np.zeros_like(direction, dtype=np.uint8)
        # 将方向量化为 0° (水平方向)
        quantized_direction[(0 <= direction) & (direction < 22.5)] = 0
        quantized_direction[(157.5 <= direction) & (direction <= 180)] = 0
        # 将方向量化为 45° (从左下到右上)
        quantized_direction[(22.5 <= direction) & (direction < 67.5)] = 45
        # 将方向量化为 90° (垂直方向)
        quantized_direction[(67.5 <= direction) & (direction < 112.5)] = 90
        # 将方向量化为 135° (从左上到右下)
        quantized_direction[(112.5 <= direction) & (direction < 157.5)] = 135

        return quantized_direction

    @staticmethod
    def apply_non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Apply Non-Maximum Suppression (NMS) to thin edges.

        Args:
            magnitude (np.ndarray): Gradient magnitude.
            direction (np.ndarray): Gradient direction in radians.

        Returns:
            np.ndarray: Thinned edge map after NMS.
        """
        # 将梯度方向量化为 4 个主要方向
        quantized_direction = NonMaximumSuppression.quantize_gradient_direction(direction)

        # 初始化一个与输入大小相同的输出数组，用于存储抑制后的结果
        suppressed_image = np.zeros_like(magnitude, dtype=np.float32)

        # 遍历图像，跳过边界像素
        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                # 获取当前像素的量化梯度方向
                current_direction = quantized_direction[i, j]

                # 根据梯度方向选择相邻像素
                if current_direction == 0:  # 水平方向
                    neighbors = (magnitude[i, j - 1], magnitude[i, j + 1])
                elif current_direction == 45:  # 对角线方向 (从左下到右上)
                    neighbors = (magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])
                elif current_direction == 90:  # 垂直方向
                    neighbors = (magnitude[i - 1, j], magnitude[i + 1, j])
                elif current_direction == 135:  # 对角线方向 (从左上到右下)
                    neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
                else:
                    neighbors = (0, 0)  # 不应发生此情况

                # 如果当前像素的梯度幅值是局部最大值，则保留该值
                if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:
                    suppressed_image[i, j] = magnitude[i, j]

        return suppressed_image

# 示例用法
if __name__ == "__main__":
    # 示例梯度幅值和方向图像
    gradient_magnitude = np.array([[0, 50, 100], [50, 150, 50], [0, 50, 0]], dtype=np.float32)
    gradient_direction = np.array([[0, np.pi/4, np.pi/2], [np.pi/4, np.pi/2, 3*np.pi/4], [np.pi, 3*np.pi/4, np.pi]], dtype=np.float32)

    # 应用非极大值抑制
    nms = NonMaximumSuppression()
    result = nms.apply_non_maximum_suppression(gradient_magnitude, gradient_direction)

    # 打印结果
    print("原始梯度幅值:")
    print(gradient_magnitude)

    print("非极大值抑制后的结果:")
    print(result)