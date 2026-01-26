import cv2
import matplotlib.pyplot as plt

def plot_cv_img(original, filtered, title1="Original", title2="Median Filtered"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis("off")

    plt.show()

def main():
    # Read image (use salt-and-pepper noisy image)
    img = cv2.imread("../data/images/salt-and-pepper.jpg")

    # Apply median filter
    # Kernel size must be odd: 3, 5, 7, ...
    median = cv2.medianBlur(img, 5)

    # Plot result
    plot_cv_img(img, median)

if __name__ == "__main__":
    main()
