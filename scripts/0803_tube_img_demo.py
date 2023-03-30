import cv2
import numpy as np


def get_tube_img(img: np.ndarray) -> np.ndarray:
    
    gray = cv2.split(img)[2] #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 4)
    ret, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_TRIANGLE)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.imshow('binary', binary)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    max_area = 0
    max_id = -1
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > max_area:
            max_area = cv2.contourArea(contour)
            max_id = i
    
    rect = cv2.minAreaRect(contours[max_id])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    cv2.rectangle(binary, box[0], box[2], (0, 255, 0))
    cv2.imshow('rect', binary)
    
    
    crop_box = np.array(
        [[0, 0],
         [0, img.shape[1]],
         [img.shape[0], img.shape[1]],
         [img.shape[0], 0]],
        dtype=np.int64        
    )
    warp_mat = cv2.getPerspectiveTransform(box.astype(np.float32), crop_box.astype(np.float32))
    crop_img = cv2.warpPerspective(img, warp_mat, img.shape[1::-1])
    
    return crop_img




img = cv2.imread(r'C:\Users\HP\Downloads\csv1_comet_1\csv1_comet_1\images\2.png')
img = cv2.resize(img, (640, 640))
tube_img =  get_tube_img(img)

cv2.imshow("img", img)
cv2.imshow("tube_img", tube_img)
cv2.waitKey()
