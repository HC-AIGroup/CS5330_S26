import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read color image (BGR)
    img = cv2.imread('../data/images/flower.jpg')

    # Custom sharpening kernel
    kernel = np.array([
        [-1/9,  -1/9,  -1/9],
        [-1/9,  2,  -1/9],
        [-1/9,  -1/9,  -1/9]
    ], dtype=np.float32)

    # Apply kernel (OpenCV handles all channels internally)
    filtered = cv2.filter2D(img, -1, kernel)

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # Show only final results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original (Color)")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Filtered (sharpened image)")
    plt.imshow(filtered_rgb)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()