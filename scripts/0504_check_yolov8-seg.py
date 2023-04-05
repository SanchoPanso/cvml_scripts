import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')
cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    results = model(img)  # list of Results objects

    img_mask = np.zeros(img.shape[:2], dtype='float32')
    for result in results:
        boxes = result.boxes
        masks = result.masks
        data = masks.data
        data = data.cpu().numpy()
        
        for i in range(data.shape[0]):
            img_mask += data[i, :, :]
    
    cv2.imshow("img", img)
    cv2.imshow("mask", img_mask)
    if cv2.waitKey() == 27:
        break



