import cv2
import numpy as np
import matplotlib.pyplot as plt


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by keeping every factor-th pixel (no resizing back)."""
    return img[::factor, ::factor]


def show_true_size_bgr(img_bgr: np.ndarray, title: str, dpi: int = 100) -> None:
    """
    Display an image at its *true pixel size* (no subplot stretching).
    - Matplotlib normally stretches images to fill the axes; this avoids that.
    """
    h, w = img_bgr.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the whole figure, no margins
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax.set_title(f"{title}  ({w}×{h})", fontsize=10, pad=6)
    ax.axis("off")
    plt.show()


def main():
    # ===== 1) Read a COLOR image =====
    img = cv2.imread("../data/images/flower.jpg")  # change path as needed
    if img is None:
        raise FileNotFoundError("Could not read image. Check the path.")

    # ===== 2) Downsample WITHOUT filtering =====
    down2_no = downsample(img, 2)
    down4_no = downsample(img, 4)

    # ===== 3) Filter (low-pass) BEFORE downsampling =====
    # Rule of thumb: stronger blur for larger downsampling
    blur2 = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1)
    blur4 = cv2.GaussianBlur(img, (9, 9), sigmaX=2, sigmaY=2)

    # ===== 4) Downsample WITH filtering =====
    down2_yes = downsample(blur2, 2)
    down4_yes = downsample(blur4, 4)

    # ===== 5) Print shapes (proves true sizes) =====
    print("Shapes (H, W, C):")
    print("Original        :", img.shape)
    print("Down ×2 no filt :", down2_no.shape)
    print("Down ×4 no filt :", down4_no.shape)
    print("Down ×2 Gaussian:", down2_yes.shape)
    print("Down ×4 Gaussian:", down4_yes.shape)

    # ===== 6) Show each image at TRUE pixel size =====
    show_true_size_bgr(img, "Original")
    show_true_size_bgr(down2_no, "Down ×2 (no filter)")
    show_true_size_bgr(down4_no, "Down ×4 (no filter)")
    show_true_size_bgr(down2_yes, "Down ×2 (Gaussian prefilter)")
    show_true_size_bgr(down4_yes, "Down ×4 (Gaussian prefilter)")


if __name__ == "__main__":
    main()
