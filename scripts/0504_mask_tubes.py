import os
import cv2
import numpy as np
from ultralytics import YOLO


src_dir = ''
dst_dir = ''
weights = 'yolov8n-seg.pt'

img_files = os.listdir(src_dir)
model = YOLO(weights)
os.makedirs(dst_dir, exist_ok=True)

for file in img_files:
    img = cv2.imread(os.path.join(src_dir, file))
    results = model(img, stream=True)
    
    img_mask = np.zeros(img.shape[:2], dtype='uint8')
    
    for result in results:
        masks = result.masks  # Masks object for segmentation masks outputs
    
    data = masks.data
    data = data.cpu().numpy()
    data *= 255
    data = data.astype('uint8')
    
    for obj_data in data:
        img_mask += obj_data
    
    cv2.imshow('img_mask', img_mask)
    cv2.waitKey()
    
    res_img = np.zeros((img.shape[0], img.shape[1], 4), dtype='uint8')
    res_img[:, :, 0:3] = img
    res_img[:, :, 3] = img_mask
    
    cv2.imwrite(os.path.join(dst_dir, file), res_img)
    print(file)
