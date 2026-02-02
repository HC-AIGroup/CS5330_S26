import cv2
import numpy as np

img = cv2.imread("../data/images/flower.jpg")  # change path
if img is None:
    raise FileNotFoundError("Could not read image.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

# Harris parameters
block_size = 2
ksize = 3
k = 0.04

R = cv2.cornerHarris(gray_f, blockSize=block_size, ksize=ksize, k=k)

R_dilated = cv2.dilate(R, None)
eps = 1e-6
local_max = (R >= (R_dilated - eps))

# ---- Threshold (start low, then increase) ----
# If you used 0.02 and got nothing, try 0.001 or 0.005
thresh = 0.005 * R.max()

corners_mask = local_max & (R > thresh)

# Debug: how many pixels survived?
print("R.max():", float(R.max()))
print("Threshold:", float(thresh))
print("Num corner pixels:", int(np.sum(corners_mask)))

out = img.copy()
out[corners_mask] = (0, 0 ,255)  # red

cv2.imshow("Harris + NMS", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
