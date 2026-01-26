import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image
img = cv2.imread("../data/images/panda.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Apply Sobel filters
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # x-derivative
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # y-derivative

# 3. Convert to displayable format; convertScaleAbs: turns signed gradient values into a displayable image.
gx_abs = cv2.convertScaleAbs(gx)
gy_abs = cv2.convertScaleAbs(gy)

# 4. Display side by side
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original (Grayscale)")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Sobel X (Vertical Edges)")
plt.imshow(gx_abs, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Sobel Y (Horizontal Edges)")
plt.imshow(gy_abs, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
