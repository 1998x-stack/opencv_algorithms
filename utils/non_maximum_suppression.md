### 非极大值抑制（Non-Maximum Suppression, NMS）

**非极大值抑制（NMS）** 是 **Canny 边缘检测** 中的重要步骤之一，其作用是**细化边缘**，通过抑制梯度方向上非最大值的像素，从而确保最终边缘更窄、更精确。它在许多图像处理和计算机视觉任务中（例如目标检测和边缘检测）被广泛应用。

---

## **1. 核心思想**

在梯度计算后，图像的梯度强度表示了边缘的强度，但边缘的宽度通常较宽（即多个像素都可能表示同一个边缘）。非极大值抑制的目的是：
1. 在梯度方向上找到局部最大值。
2. 将非局部最大值的像素设置为零，从而细化边缘。

---

## **2. 输入与输出**

- **输入**：
  - 梯度强度图（Magnitude）：由 Sobel 算子等计算得出，表示梯度的强度。
  - 梯度方向图（Direction）：表示梯度的方向（通常以弧度或度数表示）。

- **输出**：
  - 一个细化后的边缘图，保留局部最大值像素，抑制非局部最大值像素。

---

## **3. 实现步骤**

### **(1) 梯度方向量化**
梯度方向是一个连续的值（以弧度或度数表示），但在非极大值抑制中，需要将其量化为 4 个主方向：
- 0°（水平方向）
- 45°（右上方向）
- 90°（垂直方向）
- 135°（左上方向）

**量化规则**：
- 将梯度方向（弧度）转换为度数：$ \text{Direction (degrees)} = \text{Direction (radians)} \times \frac{180}{\pi} $
- 方向范围调整到 $ [0°, 180°] $：
  $$
  \text{Direction (degrees)} = (\text{Direction (degrees)} + 180) \% 180
  $$
- 根据以下范围进行量化：
  $$
  \begin{aligned}
  &\text{0°: } (0° \leq \text{Direction} < 22.5°) \text{ or } (157.5° \leq \text{Direction} \leq 180°) \\\\
  &\text{45°: } (22.5° \leq \text{Direction} < 67.5°) \\\\
  &\text{90°: } (67.5° \leq \text{Direction} < 112.5°) \\\\
  &\text{135°: } (112.5° \leq \text{Direction} < 157.5°)
  \end{aligned}
  $$

### **(2) 梯度方向的邻域比较**
根据量化后的方向，比较当前像素的梯度强度与其在梯度方向上的两个邻居像素的梯度强度：
- 如果当前像素的梯度强度小于其任一邻居，则认为它不是局部最大值，将其设置为 0。
- 如果当前像素的梯度强度是局部最大值，则保留其强度。

例如：
- 对于量化方向为 0°（水平边缘）：
  - 当前像素与其左侧和右侧的梯度强度进行比较。
- 对于量化方向为 45°（右上边缘）：
  - 当前像素与其左下方和右上方的梯度强度进行比较。
- 对于量化方向为 90°（垂直边缘）：
  - 当前像素与其上方和下方的梯度强度进行比较。
- 对于量化方向为 135°（左上边缘）：
  - 当前像素与其右下方和左上方的梯度强度进行比较。

### **(3) 更新输出图像**
对图像的每个像素重复上述操作，最终生成一个细化后的边缘图。

---

## **4. 数学表示**

设：
- 梯度强度图为 $ M(x, y) $。
- 梯度方向图为 $ \theta(x, y) $。

对于每个像素 $ (x, y) $：
1. 量化梯度方向 $ \theta(x, y) $。
2. 找到在梯度方向上的两个邻域像素 $ M_1 $ 和 $ M_2 $：
   - 根据量化方向，选择对应的邻域坐标，例如：
     - 水平方向（0°）：邻居为 $ M(x, y - 1) $ 和 $ M(x, y + 1) $。
     - 垂直方向（90°）：邻居为 $ M(x - 1, y) $ 和 $ M(x + 1, y) $。
3. 判断是否满足局部最大值条件：
   $$
   M(x, y) = 
   \begin{cases} 
   M(x, y), & \text{if } M(x, y) \geq M_1 \text{ and } M(x, y) \geq M_2 \\\\
   0, & \text{otherwise}
   \end{cases}
   $$

---

## **5. Python 实现**

以下是非极大值抑制的实现代码：

```python
import numpy as np

def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    \"\"\"Apply Non-Maximum Suppression (NMS) to thin edges.\n\n    Args:\n        magnitude (np.ndarray): Gradient magnitude.\n        direction (np.ndarray): Gradient direction (in radians).\n\n    Returns:\n        np.ndarray: Thinned edges after NMS.\n    \"\"\"\n    # 初始化输出图像\n    nms_image = np.zeros_like(magnitude, dtype=np.float32)\n\n    # 将梯度方向从弧度转换为度数并量化到 [0°, 180°]\n    direction = (direction * 180.0 / np.pi) % 180\n\n    # 遍历每个像素\n    for i in range(1, magnitude.shape[0] - 1):\n        for j in range(1, magnitude.shape[1] - 1):\n            # 获取当前像素梯度方向\n            angle = direction[i, j]\n\n            # 获取梯度方向上的邻居像素\n            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):  # 水平方向\n                neighbors = (magnitude[i, j - 1], magnitude[i, j + 1])\n            elif 22.5 <= angle < 67.5:  # 右上方向\n                neighbors = (magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])\n            elif 67.5 <= angle < 112.5:  # 垂直方向\n                neighbors = (magnitude[i - 1, j], magnitude[i + 1, j])\n            elif 112.5 <= angle < 157.5:  # 左上方向\n                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])\n\n            # 判断是否是局部最大值\n            if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:\n                nms_image[i, j] = magnitude[i, j]\n\n    return nms_image
```

---

## **6. 实验步骤**

1. **输入**：
   - 梯度强度图和方向图（可以通过 Sobel 滤波计算得到）。
2. **执行非极大值抑制**：
   - 通过上述代码细化边缘。
3. **输出**：
   - 一个更窄、更精确的边缘图。

---

## **7. 注意事项**

1. **量化梯度方向**：
   - 方向量化可能会引入误差，但它是非极大值抑制中不可避免的过程。
2. **边界处理**：
   - 对图像边界的像素，应使用零填充或跳过，以避免越界访问。
3. **计算效率**：
   - 非极大值抑制需要逐像素比较，计算量较大，可结合硬件优化（如 GPU 加速）。

---

## **8. 整体效果**

- **梯度强度图**：显示边缘的强度，但边缘宽度较宽。
- **非极大值抑制后的图像**：保留边缘像素的局部最大值，边缘变得更细、更清晰，为后续的双阈值检测和边缘追踪奠定基础。

通过非极大值抑制，边缘检测的结果会更符合实际边界的形状和位置。