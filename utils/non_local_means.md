### 非局部均值（Non-Local Means, NLM）：详细展开

非局部均值（Non-Local Means, NLM）是一种强大的图像去噪算法，与传统的局部滤波方法（如均值滤波、中值滤波）不同，**非局部均值滤波利用图像中重复的纹理或结构信息来去除噪声**，而不仅仅依赖于像素邻域的信息。它特别适合处理复杂纹理和高噪声密度的图像。

---

## **1. 核心思想**

非局部均值滤波的核心思想是：
- 图像中的噪声往往是局部随机的，但图像中的纹理和结构具有**重复性**。
- 每个像素的值可以通过与其“非局部”相似区域加权平均来估计，而不仅仅是依赖于其局部邻域。

这种方法能更好地保留图像中的细节和纹理，同时显著减少噪声。

---

## **2. 数学定义**

给定图像中的一个像素 $ p $ 及其灰度值 $ I(p) $，非局部均值滤波估算去噪后的值 $ NL(p) $：

$$
NL(p) = \frac{\sum_{q \in \Omega} w(p, q) \cdot I(q)}{\sum_{q \in \Omega} w(p, q)}
$$

其中：
- $ NL(p) $：像素 $ p $ 的去噪后的值。
- $ \Omega $：图像的像素集合（即“非局部”范围）。
- $ w(p, q) $：权重，表示像素 $ q $ 对像素 $ p $ 的相似性。
- $ I(q) $：像素 $ q $ 的原始灰度值。

### **权重的计算**

权重 $ w(p, q) $ 由像素 $ p $ 和 $ q $ 的相似性决定：
$$
w(p, q) = \exp\left(-\frac{\| I(N_p) - I(N_q) \|_2^2}{h^2}\right)
$$
- $ I(N_p) $ 和 $ I(N_q) $：分别表示像素 $ p $ 和 $ q $ 的局部邻域（称为“搜索窗口”）的灰度值。
- $ \| I(N_p) - I(N_q) \|_2^2 $：表示两个邻域之间的欧几里得距离（灰度差异）。
- $ h $：控制权重分布的参数（平滑参数）。

权重满足归一化条件：
$$
\sum_{q \in \Omega} w(p, q) = 1
$$

### **解释**
- 当两个像素的邻域非常相似时，权重 $ w(p, q) $ 较大。
- 当两个像素的邻域差异较大时，权重 $ w(p, q) $ 较小。

---

## **3. 工作流程**

非局部均值滤波的步骤如下：

1. **定义搜索窗口**：
   - 为每个像素 $ p $，定义一个大的搜索窗口 $ \Omega $，在该窗口内查找与像素 $ p $ 的邻域相似的像素。

2. **计算相似性权重**：
   - 对于搜索窗口内的每个像素 $ q $，计算其与像素 $ p $ 的相似性权重 $ w(p, q) $。

3. **加权平均**：
   - 根据权重 $ w(p, q) $，对搜索窗口内的所有像素值进行加权平均，得到去噪后的值 $ NL(p) $。

4. **重复操作**：
   - 对图像中的每个像素重复上述步骤，完成整个图像的去噪。

---

## **4. 参数设置**

非局部均值滤波中有几个关键参数：

1. **搜索窗口大小**：
   - 搜索窗口越大，能够考虑更多的非局部信息，但计算开销更高。
   - 通常设置为 $ 21 \\times 21 $ 或 $ 31 \\times 31 $。

2. **邻域大小（Patch Size）**：
   - 邻域大小决定用于计算相似性的局部区域。
   - 通常设置为 $ 7 \\times 7 $ 或 $ 5 \\times 5 $。

3. **平滑参数 $ h $**：
   - 控制相似性权重的衰减速度。
   - $ h $ 的值越大，权重越均匀；值越小，权重越集中在相似像素上。
   - 通常 $ h $ 的值与噪声强度成正比。

---

## **5. 特点**

### **优点**
1. **保留细节**：
   - 能够利用图像的全局相似性，显著减少噪声同时保留图像的边缘和细节。
2. **适应性强**：
   - 对高噪声图像和复杂纹理图像均有良好的去噪效果。
3. **无模糊效果**：
   - 与均值滤波相比，非局部均值滤波不会导致图像模糊。

### **缺点**
1. **计算复杂度高**：
   - 对每个像素需要计算其与搜索窗口内所有像素的相似性，计算量较大。
2. **参数敏感**：
   - 搜索窗口大小、邻域大小和参数 $ h $ 的选择对去噪效果有显著影响。

---

## **6. 应用场景**

1. **医学图像处理**：
   - 去除 MRI、CT 图像中的高斯噪声，同时保留细节。
2. **遥感图像**：
   - 去除卫星图像中的噪声，增强地形细节。
3. **工业图像检测**：
   - 去除工业图像中的传感器噪声，同时保持缺陷特征。
4. **艺术图像修复**：
   - 修复受噪声影响的艺术图像，恢复细节和纹理。

---

## **7. Python 实现非局部均值滤波**

以下代码展示了从零实现非局部均值滤波的过程：

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def non_local_means(image: np.ndarray, patch_size: int, search_window: int, h: float) -> np.ndarray:
    \"\"\"Apply Non-Local Means (NLM) filtering to a grayscale image.\n\n    Args:\n        image (np.ndarray): Input grayscale image.\n        patch_size (int): Size of the patch (e.g., 3x3, 5x5).\n        search_window (int): Size of the search window (e.g., 21x21).\n        h (float): Filtering parameter (controls similarity).\n\n    Returns:\n        np.ndarray: Denoised image.\n    \"\"\"\n    height, width = image.shape\n    padded_image = np.pad(image, search_window // 2, mode=\"constant\", constant_values=0)\n    denoised_image = np.zeros_like(image, dtype=np.float32)\n\n    # 遍历每个像素\n    for i in range(height):\n        for j in range(width):\n            # 获取当前像素的邻域 (patch)\n            patch_p = padded_image[i:i + patch_size, j:j + patch_size]\n\n            # 初始化权重和归一化因子\n            weights = []\n            norm_factor = 0\n\n            # 遍历搜索窗口内的像素\n            for m in range(-search_window // 2, search_window // 2 + 1):\n                for n in range(-search_window // 2, search_window // 2 + 1):\n                    neighbor_i, neighbor_j = i + m, j + n\n\n                    # 跳过越界像素\n                    if neighbor_i < 0 or neighbor_i >= height or neighbor_j < 0 or neighbor_j >= width:\n                        continue\n\n                    # 获取邻域 (patch)\n                    patch_q = padded_image[neighbor_i:neighbor_i + patch_size, neighbor_j:neighbor_j + patch_size]\n\n                    # 计算两个邻域之间的欧几里得距离\n                    distance = np.sum((patch_p - patch_q) ** 2)\n\n                    # 计算相似性权重\n                    weight = np.exp(-distance / (h ** 2))\n                    weights.append((weight, padded_image[neighbor_i + patch_size // 2, neighbor_j + patch_size // 2]))\n                    norm_factor += weight\n\n            # 计算加权均值\n            denoised_image[i, j] = sum(w * v for w, v in weights) / norm_factor\n\n    return np.clip(denoised_image, 0, 255).astype(np.uint8)\n\n# 加载灰度图像\nimage_path = \"example.jpg\"  # 替换为实际图像路径\noriginal_image = np.array(Image.open(image_path).convert(\"L\"), dtype=np.uint8)\n\n# 应用非局部均值滤波\npatch_size = 5\nsearch_window = 21\nh = 10.0\ndenoised_image = non_local_means(original_image, patch_size, search_window, h)\n\n# 显示结果\nplt.figure(figsize=(10, 5))\nplt.subplot(1, 2, 1)\nplt.imshow(original_image, cmap=\"gray\")\nplt.title(\"Original Image\")\nplt.axis(\"off\")\n\nplt.subplot(1, 2, 2)\nplt.imshow(denoised_image, cmap=\"gray\")\nplt.title(\"Denoised Image (NLM)\")\nplt.axis(\"off\")\nplt.tight_layout()\nplt.show()\n

```

---

## **8. 总结**

1. **非局部均值滤波**利用图像的全局相似性，在去噪的同时保留细节和纹理。
2. 它适用于多种场景，包括医学图像、工业图像和遥感图像处理。
3. 尽管计算复杂度较高，但优化的实现（如 OpenCV 提供的 NLM）能显著提高速度。
4. 参数设置（邻域大小、搜索窗口、平滑参数 $ h $）是影响去噪效果的关键，应根据实际应用调整。