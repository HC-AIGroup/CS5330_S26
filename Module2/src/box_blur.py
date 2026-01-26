import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_cv_img(original, filtered):
    """Display original and filtered images side by side"""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Chest Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Box Filtered Image")
    plt.imshow(filtered, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # -------------------------------------------------
    # 1. Read chest image (grayscale)
    # -------------------------------------------------
    img = cv2.imread("../data/images/chest.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Chest image not found. Check the file path.")

    # -------------------------------------------------
    # 2. Define a 3x3 box (mean) filter (normalized)
    # -------------------------------------------------
    box_filter = np.ones((5, 5), dtype=np.float32) / 9.0

    # -------------------------------------------------
    # 3. Apply box filter using filter2D
    # -------------------------------------------------
    filtered_img = cv2.filter2D(img, ddepth=-1, kernel=box_filter)

    # -------------------------------------------------
    # 4. Plot results
    # -------------------------------------------------
    plot_cv_img(img, filtered_img)


if __name__ == "__main__":
    main()
