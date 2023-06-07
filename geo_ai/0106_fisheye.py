import glob
import os
import cv2
import numpy as np

img_dir = r'C:\Users\HP\Downloads\360_5'
save_dir = r'C:\Users\HP\Downloads\360_5_separated_0606'
os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(img_dir):

    # img = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\pano_000005_000026.jpg')
    img = cv2.imread(os.path.join(img_dir, filename))
    src = img.copy()
    dst = np.zeros((2048, 2048), dtype='uint8')

    map_x = np.zeros(dst.shape[:2], dtype='float32')
    map_y = np.zeros(dst.shape[:2], dtype='float32')

    f = 1024

    width = dst.shape[1]
    height = dst.shape[0]

    x = np.arange(-width // 2, width // 2, 1, dtype='float32').reshape(1, width)
    y = np.arange(-height // 2, height // 2, 1, dtype='float32').reshape(1, height)

    pixels_per_radian = 4000 / np.pi
    name, ext = os.path.splitext(filename)
        
    for i in range(4):
        c_phi =  i * np.pi / 2
        c_theta =  np.pi / 2

        map_x_1d = (c_phi + np.arctan(x / f))
        map_x  = np.ones((height, 1), dtype='float32') @ map_x_1d * pixels_per_radian
        map_y = (c_theta + np.arctan(y.T @ (np.cos(map_x_1d - c_phi)) / f)) * pixels_per_radian

        dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        new_name = f'{name}_{i}{ext}'
        cv2.imwrite(os.path.join(save_dir, new_name), dst)
        print(new_name)
        # cv2.imshow('src', cv2.resize(src, (800, 400)))
        # cv2.imshow(f'dst {i}', cv2.resize(dst, (400, 400)))
    
    for i in range(2):
        c_phi =  np.pi
        c_theta =  i * np.pi
        
        f = 1024
        
        x_2d = np.ones((height, 1), dtype='float32') @ x
        y_2d =  y.reshape(height, 1) @ np.ones((1, width), dtype='float32')

        map_x  = (c_phi + np.arctan2(y_2d, x_2d)) * pixels_per_radian 
        map_y = (c_theta + np.arctan(np.sqrt((x_2d ** 2 + y_2d ** 2)) / f)) * pixels_per_radian

        dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        new_name = f'{name}_{i + 4}{ext}'
        cv2.imwrite(os.path.join(save_dir, new_name), dst)
        print(new_name)
        
        # cv2.imshow('src', cv2.resize(src, (800, 400)))
        # cv2.imshow(f'dst {i + 5}', cv2.resize(dst, (400, 400)))
    
    # cv2.waitKey()



