import rasterio
import numpy as np
import cv2

path = r'C:\Users\HP\Downloads\True_1255.tif'
dataset = rasterio.open(path)

print(dataset.name)
print('count:', dataset.count)
print('width:', dataset.width)
print('height:', dataset.height)
print('bounds:', dataset.bounds)
print('coordinate reference system:', dataset.crs)


for i in dataset.indexes:
    band = dataset.read(i)
    cv2.imshow(str(i), cv2.resize(band, (300, 300)))

ch4 =  dataset.read(4)
cv2.waitKey()


