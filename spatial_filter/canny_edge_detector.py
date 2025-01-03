import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from typing import Tuple

class CannyEdgeDetector:
    """
    A class to implement the Canny edge detection algorithm from scratch.
    Steps include:
    1. Gaussian filtering to remove noise.
    2. Gradient computation to calculate edge magnitude and direction.
    3. Non-maximum suppression to thin the edges.
    4. Double thresholding and edge tracking by hysteresis.
    """

    @staticmethod
    def gaussian_filtering(image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian filtering to the input image to reduce noise.

        Args:
            image (np.ndarray): Input grayscale image.
            sigma (float): Standard deviation for the Gaussian kernel.

        Returns:
            np.ndarray: Smoothed image.
        """
        return gaussian_filter(image, sigma=sigma)

    @staticmethod
    def gradient_computation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient magnitude and direction using Sobel filters.

        Args:
            image (np.ndarray): Input grayscale image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradient magnitude and gradient direction (in radians).
        """
        # Sobel kernels for x and y gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Convolve with Sobel kernels
        gradient_x = convolve(image, sobel_x)
        gradient_y = convolve(image, sobel_y)

        # Compute magnitude and direction
        magnitude = np.hypot(gradient_x, gradient_y)
        direction = np.arctan2(gradient_y, gradient_x)

        return magnitude, direction

    @staticmethod
    def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Apply non-maximum suppression to thin the edges.

        Args:
            magnitude (np.ndarray): Gradient magnitude.
            direction (np.ndarray): Gradient direction (in radians).

        Returns:
            np.ndarray: Thinned edges.
        """
        # Quantize gradient directions to 4 main directions (0, 45, 90, 135 degrees)
        direction = direction * (180.0 / np.pi)  # Convert to degrees
        direction = (direction + 180) % 180      # Ensure all angles are positive

        thinned_image = np.zeros_like(magnitude, dtype=np.float32)

        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                # Determine the neighboring pixels in the direction of the gradient
                angle = direction[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = (magnitude[i, j - 1], magnitude[i, j + 1])
                elif 22.5 <= angle < 67.5:
                    neighbors = (magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])
                elif 67.5 <= angle < 112.5:
                    neighbors = (magnitude[i - 1, j], magnitude[i + 1, j])
                elif 112.5 <= angle < 157.5:
                    neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

                # Suppress non-maximum pixels
                if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:
                    thinned_image[i, j] = magnitude[i, j]

        return thinned_image

    @staticmethod
    def double_thresholding(image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
        """
        Apply double thresholding to classify edges as strong, weak, or non-edges.

        Args:
            image (np.ndarray): Input thinned edge image.
            low_threshold (float): Low threshold value.
            high_threshold (float): High threshold value.

        Returns:
            np.ndarray: Binary edge map with strong and weak edges.
        """
        strong_edges = (image >= high_threshold).astype(np.uint8)
        weak_edges = ((image >= low_threshold) & (image < high_threshold)).astype(np.uint8)
        return strong_edges, weak_edges

    @staticmethod
    def edge_tracking_by_hysteresis(strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
        """
        Perform edge tracking by hysteresis to connect weak edges to strong edges.

        Args:
            strong_edges (np.ndarray): Binary map of strong edges.
            weak_edges (np.ndarray): Binary map of weak edges.

        Returns:
            np.ndarray: Final edge map.
        """
        final_edges = strong_edges.copy()
        height, width = strong_edges.shape

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if weak_edges[i, j] and np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    final_edges[i, j] = 1

        return final_edges

    def canny_edge_detection(self, image: np.ndarray, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
        """
        Perform the full Canny edge detection algorithm.

        Args:
            image (np.ndarray): Input grayscale image.
            sigma (float): Standard deviation for Gaussian smoothing.
            low_threshold (float): Low threshold for double thresholding.
            high_threshold (float): High threshold for double thresholding.

        Returns:
            np.ndarray: Final edge map.
        """
        # Step 1: Gaussian filtering
        smoothed_image = self.gaussian_filtering(image, sigma)

        # Step 2: Gradient computation
        magnitude, direction = self.gradient_computation(smoothed_image)

        # Step 3: Non-maximum suppression
        thinned_edges = self.non_maximum_suppression(magnitude, direction)

        # Step 4: Double thresholding
        strong_edges, weak_edges = self.double_thresholding(thinned_edges, low_threshold, high_threshold)

        # Step 5: Edge tracking by hysteresis
        final_edges = self.edge_tracking_by_hysteresis(strong_edges, weak_edges)

        return final_edges

# 工业场景示例
if __name__ == "__main__":
    # 加载灰度图像
    image_path = "example.jpg"  # 替换为实际图像路径
    original_image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)

    # 实例化 CannyEdgeDetector
    canny_detector = CannyEdgeDetector()

    # 应用 Canny 边缘检测
    print("Applying Canny Edge Detection...")
    final_edges = canny_detector.canny_edge_detection(original_image, sigma=1.0, low_threshold=20, high_threshold=40)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(final_edges, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
