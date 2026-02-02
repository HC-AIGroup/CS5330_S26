import cv2

img = cv2.imread("../data/images/flower.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not read img.jpg")

# SIFT detector/descriptor
sift = cv2.SIFT_create()

kps, desc = sift.detectAndCompute(img, None)
print("SIFT keypoints:", len(kps), "descriptor shape:", None if desc is None else desc.shape)

# Visualize
vis = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT keypoints", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
