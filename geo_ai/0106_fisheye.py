import cv2
import numpy as np

img = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\pano_000005_000026.jpg')
src = img[0:4000, 0:1600, :].copy()
dst = img[0:4000, 0:1600, :]

map_x = np.zeros(dst.shape[:2], dtype='float32')
map_y = np.zeros(dst.shape[:2], dtype='float32')

fx = 1024
fy = 1024

width = dst.shape[1]
height = dst.shape[0]

cx = width / 2
cy = height / 2

x = np.arange(0, width, 1, dtype='float32')
y = np.arange(0, height, 1, dtype='float32')

map_x = cx + np.arctan((x - cx) / (2 * fx)) * 2 / np.pi * 4000
map_y = cy + np.arctan((y - cy) / (2 * fy)) * 2 / np.pi * 4000


map_x = np.concatenate([map_x.reshape(1, -1)] * height, axis=0)
map_y = np.concatenate([map_y.reshape(1, -1)] * width, axis=0).T

dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

cv2.imshow('src', cv2.resize(src, (400, 800)))
cv2.imshow('dst', cv2.resize(dst, (400, 400)))
cv2.waitKey()



