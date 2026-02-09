import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(title: str, img: np.ndarray, cmap=None):
    """Matplotlib display helper (works well in PyCharm)."""
    plt.figure(figsize=(7, 5))
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap if cmap else "gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        # OpenCV uses BGR; matplotlib expects RGB
        plt.imshow(img[:, :, ::-1], interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.show()


# ----------------------------
# 1) Load image
# ----------------------------
path = "../data/panda.png"  # <-- change this
img_bgr = cv2.imread(path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image at: {path}")

# ----------------------------
# 2) Preprocess (grayscale + denoise)
# ----------------------------
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Gaussian blur is a common baseline; median is better for salt-and-pepper noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# ----------------------------
# 3) Threshold to binary
#    Option A: Otsu (good for fairly uniform lighting)
# ----------------------------
_, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# If your objects are dark on bright background, invert:
binary_otsu = cv2.bitwise_not(binary_otsu)
# ----------------------------
# 4) Morphology cleanup (remove small noise, close small gaps)
# ----------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
# ----------------------------
# 5) Find contours (treat each contour as one object)
# ----------------------------
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours:", len(contours))
for i, c in enumerate(contours[:3]):
    print(f"Contour {i} shape:", c.shape)
# ----------------------------
# 6) Draw results (contours + bounding boxes + centroids)
# ----------------------------
# Assume: contours already computed

print("Number of contours:", len(contours))

contour_img = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

cv2.drawContours(
    contour_img,
    contours,
    -1,
    (255, 255, 255),  # white contours
    2
)

plt.imshow(contour_img)
plt.title("Contours Only")
plt.axis("off")
plt.show()
# Show


# ----------------------------
# 7) Display
# ----------------------------
show("Original (BGR)", img_bgr)

