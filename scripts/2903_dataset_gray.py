import os
import cv2
import numpy as np

image_dir = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023\valid\images'
gray_dir = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023\valid_gray\images'

os.makedirs(gray_dir, exist_ok=True)
image_files = os.listdir(image_dir)

for file in image_files:
    img_path = os.path.join(image_dir, file)
    img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
    gray = img[:, :, 2]
    save_path = os.path.join(gray_dir, file)
    
    ext = os.path.splitext(os.path.split(save_path)[-1])[1]
    is_success, im_buf_arr = cv2.imencode(ext, gray)
    im_buf_arr.tofile(save_path)
    
    print(file)
    
