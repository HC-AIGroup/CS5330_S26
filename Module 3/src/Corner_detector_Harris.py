import cv2
import numpy as np

img = cv2.imread("../data/images/flower.jpg")
if img is None:
    raise FileNotFoundError("Could not read image. Check path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

# Harris response
R = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)

# --- Robust NMS: allow tiny tolerance instead of strict equality ---
R_dil = cv2.dilate(R, None)           # local max in 3x3 neighborhood
eps = 1e-6                            # tolerance for float comparisons
local_max = (R >= (R_dil - eps))      # <- key fix

# --- Threshold (start LOW) ---
thresh = 0.001 * R.max()              # try 0.001, 0.003, 0.005, 0.01
mask = local_max & (R > thresh)

print("R.max =", float(R.max()))
print("thresh =", float(thresh))
print("NMS+thresh count =", int(mask.sum()))

# Visualize as circles (much easier to see than coloring pixels)
out = img.copy()
ys, xs = np.where(mask)
for x, y in zip(xs, ys):
    cv2.circle(out, (x, y), 3, (0, 0, 255), 1)

cv2.imshow("Harris corners ", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
