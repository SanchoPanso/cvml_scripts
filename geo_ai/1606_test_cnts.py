import cv2
import numpy as np


img = np.zeros((400, 400), dtype='uint8')
cv2.circle(img, (195, 200), 50, 255, -1)
cv2.circle(img, (300, 200), 50, 255, -1)
cv2.line(img, (100, 200), (300, 200), 255, 1)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 1)

cv2.imshow('img', img)
cv2.waitKey()
