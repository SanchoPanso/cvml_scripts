import rasterio
import numpy as np
import cv2

path = r'C:\Users\HP\Downloads\3863.tif'
dataset = rasterio.open(path)

print(dataset.name)
print('count:', dataset.count)
print('width:', dataset.width)
print('height:', dataset.height)
print('bounds:', dataset.bounds)
print('coordinate reference system:', dataset.crs)


bands = []
for i in [3, 2, 1]: # dataset.indexes:
    print(i)
    band = dataset.read(i)
    bands.append(band)
    # cv2.imshow(str(i), cv2.resize(band, (300, 300)))

img = cv2.merge(bands)
cv2.imshow('img', cv2.resize(img, (300, 300)))
cv2.waitKey()

cv2.imwrite(r'C:\Users\HP\Downloads\3863.jpg', img)


