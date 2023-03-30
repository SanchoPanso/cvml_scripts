import os
import cv2
import glob
import numpy as np


img_dir = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023\valid\images'
for img_file in glob.glob(os.path.join(img_dir, '*')):
    img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('img', img)
    chs = cv2.split(img)
    for i, ch in enumerate(chs):
        cv2.imshow(str(i), ch)
    cv2.waitKey()
    
