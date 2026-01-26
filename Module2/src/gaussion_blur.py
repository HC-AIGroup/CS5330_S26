import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_cv_img(original, filtered):
    """Display original and filtered images side by side"""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original  Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("gaussion Filtered Image")
    plt.imshow(filtered, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    # read an image
    img = cv2.imread('../data/images/portrait1.jpg')

    # apply Gaussian filter
    # (kernel size = 5x5, sigma = 0 -> OpenCV chooses sigma automatically)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # do plot
    plot_cv_img(img, blur)


if __name__ == '__main__':
    main()
