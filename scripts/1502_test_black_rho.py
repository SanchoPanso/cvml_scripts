import os
import sys
import cv2

from cvml.dataset.image_transforming import convert_to_mixed, expo


images_dir = r'C:\Users\HP\Downloads\csv1_comet_1\csv1_comet_1\images'
image_files = os.listdir(images_dir)

for file in image_files:
    path = os.path.join(images_dir, file)
    img = cv2.imread(path)
    img = convert_to_mixed(img)
    phi, rho, gray = cv2.split(img)
    
    cv2.imshow('gray', cv2.resize(gray, (400, 400)))
    cv2.imshow('rho', cv2.resize(rho, (400, 400)))
    cv2.imshow('phi', cv2.resize(phi, (400, 400)))
    cv2.waitKey()
