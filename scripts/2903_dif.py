import os
import cv2
import numpy as np

def get_img_dif(img1: np.ndarray, img2: np.ndarray) -> int:
    img1 = img1.astype('int16')
    img2 = img2.astype('int16')
    dif = np.abs(img1 - img2).sum()
    
    return dif


if __name__ == '__main__':
    orig_images_dir = r'C:\Users\HP\Downloads\png_orig\save 1'
    compressed_images_dir = r'C:\Users\HP\Downloads\png_compressed(jpg)\png_compressed(jpg)'
    
    orig_images_files = os.listdir(orig_images_dir)
    compressed_images_files = os.listdir(compressed_images_dir)
    
    for compressed_file in compressed_images_files:
        print(compressed_file, '\n')
        compressed_img = cv2.imread(os.path.join(compressed_images_dir, compressed_file))
        min_dif = -1
        min_file = ''
        
        for orig_file in orig_images_files:
            orig_img = cv2.imread(os.path.join(orig_images_dir, orig_file))
            
            orig_img = cv2.resize(orig_img, (640, 535))
            compressed_img = cv2.resize(compressed_img, (640, 535))
            dif = get_img_dif(orig_img, compressed_img)
            
            print(orig_file, dif)
            if dif < min_dif or min_dif == -1:
                min_dif = dif
                min_file = orig_file
            
        print('compressed_file:', compressed_file, '     orig_file:', min_file)
        print(min_dif)
        print()



