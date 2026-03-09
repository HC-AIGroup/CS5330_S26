import torch
import cv2
import numpy as np

'''print("torch:", torch.__version__)
print("opencv:", cv2.__version__)

img = np.zeros((128,128,3), dtype=np.uint8)
tensor = torch.from_numpy(img)

print("tensor shape:", tensor.shape)

import torch
import torch.nn as nn

model = nn.Conv2d(3, 16, kernel_size=3)

x = torch.randn(1,3,128,128)
y = model(x)

print(y.shape)'''



x = torch.tensor([[1,2],[3,4]])
print(x, type)