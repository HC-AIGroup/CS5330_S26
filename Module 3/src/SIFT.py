
import cv2
import numpy as np

# Read images (grayscale)
img1 = cv2.imread("../data/images/img1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../data/images/img2.png", cv2.IMREAD_GRAYSCALE)
assert img1 is not None, "img1 not loaded"
assert img2 is not None, "img2 not loaded"
# Create SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# des1, des2 are the descriptors and NumPy array
# kp1: the keypoints: kp1 is a Python list
#kp1[0].size:the diameter of the region around the keypoint:Directly tied to scale-space detection

print("Descriptor shape:", des1.shape)
print("kp1: ( location, size, angle)",kp1[0].pt,kp1[0].size,kp1[0].angle)

#2:Measure similarity using L2 distance (Brute Force)
# Brute-Force matcher with L2 distance (required for SIFT)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Find the two nearest neighbors for each descriptor
matches_knn = bf.knnMatch(des1, des2, k=2)
#3:Apply Lowe’s Ratio Test (remove ambiguous matches)
good_matches = []
ratio_thresh = 0.75

for m, n in matches_knn:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
#4 visualize matches
matched_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("SIFT matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
