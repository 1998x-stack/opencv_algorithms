import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class AdaptiveHistogramEqualizer:
    """
    A class to perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on grayscale images.
    """

    def __init__(self, grid_size: int = 8, clip_limit: float = 0.01):
        """
        Initialize the CLAHE parameters.

        Args:
            grid_size (int): The size of each grid (e.g., 8x8).
            clip_limit (float): The contrast clip limit (proportion of total pixels in a grid).
        """
        self.grid_size = grid_size
        self.clip_limit = clip_limit

    def calculate_histogram(self, block: np.ndarray) -> np.ndarray:
        """
        Calculate the histogram for a given block.

        Args:
            block (np.ndarray): A 2D numpy array representing the block.

        Returns:
            np.ndarray: The histogram of the block with 256 bins.
        """
        histogram, _ = np.histogram(block.flatten(), bins=256, range=(0, 256))
        return histogram

    def clip_histogram(self, histogram: np.ndarray, total_pixels: int) -> np.ndarray:
        """
        Clip the histogram to limit contrast enhancement.

        Args:
            histogram (np.ndarray): The histogram of a block.
            total_pixels (int): Total number of pixels in the block.

        Returns:
            np.ndarray: The clipped histogram.
        """
        clip_limit = int(self.clip_limit * total_pixels)
        excess = np.maximum(histogram - clip_limit, 0).sum()
        histogram = np.minimum(histogram, clip_limit)
        histogram += excess // 256  # Redistribute excess uniformly
        return histogram

    def calculate_cdf(self, histogram: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function (CDF) for the histogram.

        Args:
            histogram (np.ndarray): The clipped histogram.

        Returns:
            np.ndarray: The normalized CDF mapped to the range [0, 255].
        """
        cdf = np.cumsum(histogram)
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
        return cdf_normalized.astype(np.uint8)

    def apply_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the input grayscale image.

        Args:
            image (np.ndarray): Input grayscale image as a 2D numpy array.

        Returns:
            np.ndarray: CLAHE-enhanced grayscale image.
        """
        height, width = image.shape
        grid_h = height // self.grid_size
        grid_w = width // self.grid_size

        # Initialize output image
        output = np.zeros_like(image, dtype=np.uint8)

        # Process each grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract the block
                block = image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]

                # Calculate histogram and clip it
                histogram = self.calculate_histogram(block)
                clipped_histogram = self.clip_histogram(histogram, block.size)

                # Calculate CDF and apply equalization
                cdf = self.calculate_cdf(clipped_histogram)
                block_equalized = cdf[block]

                # Place equalized block in output image
                output[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w] = block_equalized

        return output

# Industrial application example
if __name__ == "__main__":
    # Load the grayscale image using PIL
    image_path = "images/CIFAR-10-images/train/automobile/0002.jpg"  # Replace with your image path
    pil_image = Image.open(image_path).convert("L")
    input_image = np.array(pil_image, dtype=np.uint8)

    # Display original image
    print("Displaying original image...")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Perform CLAHE
    clahe = AdaptiveHistogramEqualizer(grid_size=8, clip_limit=0.01)
    enhanced_image = clahe.apply_equalization(input_image)

    # Display enhanced image
    print("Displaying enhanced image...")
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap="gray")
    plt.title("CLAHE Enhanced Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
