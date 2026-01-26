# Made by  Dr. Bayat
'''OpenCV provides the cv2.resize() function to change the size of an image.
Resizing is commonly used for preprocessing, visualization,
and preparing images for computer vision algorithms.'''
import cv2
import numpy as np
from pathlib import Path

# Part I: Loading Images
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "panda.png"

# Read image
image = cv2.imread(str(IMAGE_PATH))
# Load color image
#img = cv2.imread("/Users/kadykady/PycharmProjects/opencv-hw1/data/images/panda.png")

# Load grayscale image
gray_img = cv2.imread("../data/images/panda.png", cv2.IMREAD_GRAYSCALE)

# Check if image loaded correctly
if image is None:
    raise FileNotFoundError("Could not load panda.png. Check the path.")

height, width = image.shape[:2]

print("Width:", width)
print("Height:", height)

# Resize image (width, height)
resized_img = cv2.resize(image, (500, 500))

# Display image
cv2.imshow("Resized panda Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

''' -----Aspect ratio is preserved
cv2.resize(
    img,
    (0, 0),            # dsize ignored
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_AREA
)
#INTER_LINEAR is better for enlarging

#INTER_AREA is better for reducing'''