import cv2
import numpy as np

bgr = cv2.imread(r'D:\datasets\tmk\crops\0712_comet_crops\comet\1.jpg')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ret, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# First create the image with alpha channel
bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
# bgra = np.ones((*bgr.shape[0:2], 4),  dtype=bgr.dtype) * 255

# Then assign the mask to the last channel of the image
bgra[:, :, 3] = alpha

cv2.imwrite(r'D:\datasets\tmk\crops\0712_comet_crops\comet\1_bgra.png', bgra)
bgra = cv2.imread(r'D:\datasets\tmk\crops\0712_comet_crops\comet\1_bgra.png', cv2.IMREAD_UNCHANGED)
print(bgra.shape)

