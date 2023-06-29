import os
import cv2
import numpy as np

img_path = r'C:\Users\HP\Downloads\AAM DevelopedArea 30cm_7_4.jpg'
crop_h = 320
crop_w = 320

dst_images_dir = r'C:\Users\HP\Downloads\sat_320_1506'
os.makedirs(dst_images_dir, exist_ok=True)

img = cv2.imread(img_path)

img_h, img_w = img.shape[:2]
num_of_rows = img_h // crop_h if img_h % crop_h == 0 else img_h // crop_h + 1
num_of_cols = img_w // crop_w if img_w % crop_w == 0 else img_w // crop_w + 1
crop_cnt = 1

for row in range(num_of_rows):
    for col in range(num_of_cols):
        
        crop_x1 = crop_w * col
        crop_y1 = crop_h * row

        crop_x2 = min(crop_x1 + crop_w, img_w)
        crop_y2 = min(crop_y1 + crop_h, img_h)
        
        img_crop = img[crop_y1: crop_y2, crop_x1: crop_x2]
        cv2.imwrite(os.path.join(dst_images_dir, f'{os.path.splitext(os.path.split(img_path)[-1])[0]}_{crop_cnt}.png'), img_crop)
        
        print(crop_cnt)
        crop_cnt += 1
        