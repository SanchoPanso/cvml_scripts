import numpy as np
import cv2
from PIL import Image, ImageChops, ImageStat

img1_path = r'C:\Users\HP\Downloads\100.png'
img2_path = r'C:\Users\HP\Downloads\124.png'
img3_path = r'C:\Users\HP\Downloads\102.png'
img4_path = r'C:\Users\HP\Downloads\102.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)
img4 = cv2.imread(img4_path)

img1 = cv2.resize(img1, img4.shape[:2][::-1])
img2 = cv2.resize(img2, img4.shape[:2][::-1])
img3 = cv2.resize(img3, img4.shape[:2][::-1])

img1 = img1.astype('int16')
img2 = img2.astype('int16')
img3 = img3.astype('int16')
img4 = img4.astype('int16')

dif14 = np.abs(img1 - img4).sum()
dif24 = np.abs(img2 - img4).sum()
dif34 = np.abs(img3 - img4).sum()

print('100.png:', dif14)
print('124.png:', dif24)
print('102.png:', dif34)



# img1 = Image.open(img1_path)
# img2 = Image.open(img2_path)
# img3 = Image.open(img3_path)

# stat = ImageStat.Stat(ImageChops.difference(img1, img2))
# dif1 = stat.sum
# stat = ImageStat.Stat(ImageChops.difference(img3, img2))
# dif2 = stat.sum
# stat = ImageStat.Stat(ImageChops.difference(img3, img1))
# dif3 = stat.sum

# print(sum(dif1), sum(dif2), sum(dif3))

