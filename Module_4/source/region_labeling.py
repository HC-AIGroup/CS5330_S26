import cv2
import numpy as np
import matplotlib.pyplot as plt

binary = np.array([
    [0,0,0,0,0,0],
    [0,255,255,0,0,0],
    [0,255,255,0,0,255],
    [0,0,0,0,0,255],
    [0,0,0,0,0,0]
], dtype=np.uint8)

print("binary[0]:", binary[0])

plt.imshow(binary, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
plt.title("Binary image")
plt.axis("off")
plt.show()



num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary,
    connectivity=8
)

# Convert to color image for drawing
output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# Loop over foreground components ONLY (skip label 0)
for label in range(1, num_labels):  # label 0 = background
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    print("x,y,w,h",x,y,w,h)
    cv2.rectangle(
        output,
        (x, y),
        (x + w-1, y + h-1),
        (0, 0, 255),  # red box
        0             # thickness (NOT -1)
    )
# display correctly
    plt.imshow(output[:, :, ::-1], interpolation="nearest")  # BGR → RGB
    plt.title("Bounding Boxes")
    plt.axis("off")
    plt.show()