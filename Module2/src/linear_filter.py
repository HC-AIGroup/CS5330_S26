import cv2
import numpy as np
import matplotlib.pyplot as plt

def point_operation(gray: np.ndarray, K: float, L: float) -> np.ndarray:

    if gray.ndim != 2:
        raise ValueError("point_operation expects a grayscale image with shape (H, W).")
    # Convert to float for computation to avoid overflow/underflow
    out = gray.astype(np.float32) * K + L
    # Clip to valid grayscale range and convert back to uint8
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def main():
    # 1) Read image (BGR in OpenCV)
    img = cv2.imread("../data/images/panda.png")
    if img is None:
        raise FileNotFoundError("Could not read '../data/images/panda.png'. Check the path.")
    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3) Apply different (K, L) settings
    out1 = point_operation(gray, K=0.5, L=0)     # darker (reduced contrast)
    out2 = point_operation(gray, K=1.0, L=10)    # slightly brighter
    out3 = point_operation(gray, K=0.7, L=25)    # brighten + reduce contrast
    # 4) Display results side-by-side
    result = np.hstack([gray, out1, out2, out3])

    plt.figure(figsize=(12, 4))
    plt.imshow(result, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Original | K=0.5,L=0 | K=1.0,L=10 | K=0.7,L=25")
    plt.show()


if __name__ == "__main__":
    main()
