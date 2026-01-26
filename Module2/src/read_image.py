import cv2
from pathlib import Path

# Build path safely (works on Mac, Windows, Linux)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "panda.png"

# Read image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

# Show image
cv2.imshow("My First OpenCV Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()