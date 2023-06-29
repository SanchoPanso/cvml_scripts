from ultralytics import YOLO
import cv2

model = YOLO(r'C:\Users\HP\Downloads\yolov8s_seg_geoai360_07062023_2_best.pt')
results = model(r'D:\geo_ai_data\shots\5\pano_000005_000025_0.jpg')

for mask in results[0].masks.data:
    mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (2048, 2048)) * 255
    
    cv2.imshow('mask', cv2.resize(mask, (400, 400)))
    p = cv2.waitKey()
    
    if p == 27:
        cv2.imwrite('mask_example.png', mask)
        break
    else:
        continue
    