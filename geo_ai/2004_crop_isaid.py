import os
import cv2
import numpy as np
import cvml
from cvml.annotation.bounding_box import BoundingBox, BBType, BBFormat


crop_h = 640
crop_w = 640

src_dataset_dir = '/mnt/data/geo_ai_datasets/isaid_small'
dst_dataset_dir = '/mnt/data/geo_ai_datasets/isaid_small_cropped'

for split in ['train', 'valid']:
    src_images_dir = os.path.join(src_dataset_dir, split, 'images')
    src_annotations_dir = os.path.join(src_dataset_dir, split, 'annotations')
    src_annotation_path = os.path.join(src_annotations_dir, f'{split}.json')

    dst_images_dir = os.path.join(dst_dataset_dir, split, 'images')
    dst_annotations_dir = os.path.join(dst_dataset_dir, split, 'annotations')
    dst_annotation_path = os.path.join(dst_annotations_dir, f'{split}.json')

    os.makedirs(dst_annotations_dir, exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)

    src_annot = cvml.read_coco(src_annotation_path)
    dst_annot = cvml.Annotation(src_annot.classes, {})
    
    for img_name in src_annot.bbox_map:
        bboxes = src_annot.bbox_map[img_name]
        if len(bboxes) == 0:
            continue
        
        img = cv2.imread(os.path.join(src_images_dir, img_name + '.png'))
        img_h, img_w = img.shape[:2]
        num_of_rows = img_h // crop_h if img_h % crop_h == 0 else img_h // crop_h + 1
        num_of_cols = img_w // crop_w if img_w % crop_w == 0 else img_w // crop_w + 1
        crop_cnt = 1

        for row in range(num_of_rows):
            for col in range(num_of_cols):
                new_bboxes = []
                
                crop_x1 = crop_w * col
                crop_y1 = crop_h * row

                crop_x2 = min(crop_x1 + crop_w, img_w)
                crop_y2 = min(crop_y1 + crop_h, img_h)

                for bbox in bboxes:
                    x, y, w, h = bbox.get_coordinates()

                    if x >= crop_x2 or y >= crop_y2:
                        continue
                    if x + w <= crop_x1 or y + h <= crop_y1:
                        continue

                    x = max(x, crop_x1)
                    y = max(y, crop_y1)

                    x2 = min(x + w, crop_x2)
                    y2 = min(y + h, crop_y2)

                    x -= crop_x1
                    y -= crop_y1
                    w -= crop_x1
                    h -= crop_y1

                    segmentation = bbox.get_segmentation()
                    if len(segmentation) == 0:
                        continue
                    
                    x_points = segmentation[0][0::2]
                    y_points = segmentation[0][1::2]

                    x_points = [min(crop_x2, max(crop_x1, xp)) - crop_x1 for xp in x_points]
                    y_points = [min(crop_y2, max(crop_y1, yp)) - crop_y1 for yp in y_points]
                    
                    contour = np.array([x_points, y_points]).T
                    if cv2.contourArea(contour) < 1:
                        continue

                    segmentation[0][0::2] = x_points
                    segmentation[0][1::2] = y_points
                    
                    
                    new_bbox = BoundingBox(bbox.get_class_id(), x, y, x2 - x, y2 - y, 1, 
                                           f'{img_name}_{crop_cnt}', 
                                           img_size=(crop_x2 - crop_x1, crop_y2 - crop_y1),
                                           segmentation=segmentation)
                    new_bboxes.append(new_bbox)
                
                img_crop = img[crop_y1: crop_y2, crop_x1: crop_x2]
                #cv2.imwrite(os.path.join(dst_images_dir, f'{img_name}_{crop_cnt}.png'), img_crop)
                print(f'{img_name}_{crop_cnt}.png')
                dst_annot.bbox_map[f'{img_name}_{crop_cnt}'] = new_bboxes
                crop_cnt += 1
    
    cvml.write_coco(dst_annot, dst_annotation_path, '.png')
    os.makedirs(os.path.join(dst_dataset_dir, split, 'labels'), exist_ok=True)
    cvml.write_yolo_seg(dst_annot, os.path.join(dst_dataset_dir, split, 'labels'))

                

    
