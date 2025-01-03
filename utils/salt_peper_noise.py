import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def add_salt_and_pepper_noise(image: np.ndarray, noise_density: float) -> np.ndarray:
    """Add salt-and-pepper noise to a grayscale image."""
    noisy_image = image.copy()
    num_pixels = image.size
    num_salt = int(noise_density * num_pixels / 2)
    num_pepper = int(noise_density * num_pixels / 2)
    
    # 添加盐噪声
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # 添加椒噪声
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a median filter to a grayscale image."""
    padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.median(region)
    
    return filtered_image

# 加载灰度图像
image_path = "example.jpg"  # 替换为实际图像路径
original_image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

# 添加椒盐噪声
noise_density = 0.05
noisy_image = add_salt_and_pepper_noise(original_image, noise_density)

# 应用中值滤波去噪
denoised_image = median_filter(noisy_image, kernel_size=3)

# 可视化结果
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image (Salt-and-Pepper)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised_image, cmap="gray")
plt.title("Denoised Image (Median Filter)")
plt.axis("off")

plt.tight_layout()
plt.show()