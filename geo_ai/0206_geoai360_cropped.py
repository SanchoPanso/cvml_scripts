import os
import shutil
import cv2
import numpy as np
import cvml
from cvml.annotation.bounding_box import BoundingBox, BBType, BBFormat
import zipfile
from pathlib import Path
from filesplit.split import Split


def main():

    block_volume = 999*1024*1024
    orig_dataset_dir = r'/mnt/data/geo_ai_datasets/geoai360/geoai_360_02062023'
    cropped_dataset_dir = r'/mnt/data/geo_ai_datasets/geoai360/geoai_360_02062023_cropped_1280'
    splitted_dataset_dir = r'/mnt/data/geo_ai_datasets/geoai360/geoai_360_02062023_splitted'

    # create_yolo_labels(orig_dataset_dir)
    create_cropped_dataset(orig_dataset_dir, cropped_dataset_dir)
    create_splitted_dataset(cropped_dataset_dir, splitted_dataset_dir, block_volume)


def create_yolo_labels(dataset_dir: str):
    for split in ['train']:
        coco_path = os.path.join(dataset_dir, split, 'annotations', 'data.json')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        annot = cvml.read_coco(coco_path)
        os.makedirs(labels_dir, exist_ok=True)
        cvml.write_yolo_seg(annot, labels_dir)


def create_small_dataset(src_dir: str, dst_dir: str, percentage: float):

    for split in ['train']:
        
        os.makedirs(os.path.join(dst_dir, split, 'annotations'), exist_ok=True)
        shutil.copy(os.path.join(src_dir, split, 'annotations', f'{split}.json'), 
                    os.path.join(dst_dir, split, 'annotations', f'{split}.json'))
        
        for data_type in ['images', 'labels']:
            os.makedirs(os.path.join(dst_dir, split, data_type), exist_ok=True)
            src_files = os.listdir(os.path.join(src_dir, split, data_type))
            src_files.sort()
            chosen_src_files = src_files[:int(len(src_files) * percentage)]

            
            for f in chosen_src_files:
                src_path = os.path.join(src_dir, split, data_type, f)
                dst_path = os.path.join(dst_dir, split, data_type, f)
                shutil.copy(src_path, dst_path)
                print(dst_path)       
                
                
    # classes = ["storage_tank",
    #             "Large_Vehicle",
    #             "Small_Vehicle",
    #             "ship",
    #             "Harbor",
    #             "baseball_diamond",
    #             "Ground_Track_Field",
    #             "Soccer_ball_field",
    #             "Swimming_pool",
    #             "Roundabout",
    #             "tennis_court",
    #             "basketball_court",
    #             "plane",
    #             "Helicopter",
    #             "Bridge"]

    # description = f"train: /content/isaid/train/images\n"\
    #             f"val: /content/isaid/valid/images\n"\
    #             f"nc: {len(classes)}\n"\
    #             f"names: {str(classes)}\n"
                
    # print(description)
    # with open(os.path.join(dst_dir, 'data.yaml'), 'w') as f:
    #     f.write(description)



def create_cropped_dataset(src_dir: str, dst_dir: str, imgsz: tuple = (1280, 1280)):

    crop_h, crop_w = imgsz

    for split in ['train']:
        src_images_dir = os.path.join(src_dir, split, 'images')
        src_annotations_dir = os.path.join(src_dir, split, 'annotations')
        src_annotation_path = os.path.join(src_annotations_dir, 'data.json')

        dst_images_dir = os.path.join(dst_dir, split, 'images')
        dst_annotations_dir = os.path.join(dst_dir, split, 'annotations')
        dst_annotation_path = os.path.join(dst_annotations_dir, 'data.json')

        os.makedirs(dst_annotations_dir, exist_ok=True)
        os.makedirs(dst_images_dir, exist_ok=True)

        src_annot = cvml.read_coco(src_annotation_path)
        dst_annot = cvml.Annotation(src_annot.classes, {})
        
        for img_name in src_annot.bbox_map:
            bboxes = src_annot.bbox_map[img_name]
            if len(bboxes) == 0:
                continue
            
            img_path = os.path.join(src_images_dir, img_name + '.jpg')
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
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
                        mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype='uint8')
                        
                        segmentation = bbox.get_segmentation()
                        cls_id = bbox.get_class_id()
                        
                        if len(segmentation) == 0:
                            continue
                        
                        s = segmentation[0]
                        s_copy = []
                        for i in range(len(s)):
                            s_copy.append(s[i] - crop_x1 if i%2==0 else s[i] - crop_y1)
                        pts = np.array(s_copy, dtype='int32')
                        pts = pts.reshape((-1, 1, 2))
                        mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                                      
                        if mask.max() == 0:
                            continue

                        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                        if len(contours) == 0:
                            continue

                        c = contours[0]
                        new_segmentation = c.reshape(1, -1).tolist()
                        x_pts = new_segmentation[0][0::2]
                        y_pts = new_segmentation[0][1::2]
                        
                        x = max(x_pts)
                        y = max(y_pts)
                        x2 = min(x_pts)
                        y2 = min(y_pts)
                        
                        w = x2 - x
                        h = y2 - y
                            
                        new_bbox = BoundingBox(cls_id, 
                                                x, y, x2 - x, y2 - y, 1, 
                                                f'{img_name}_{crop_cnt}', 
                                                img_size=(crop_x2 - crop_x1, crop_y2 - crop_y1),
                                                segmentation=new_segmentation)
                        new_bboxes.append(new_bbox)
                
                    img_crop = img[crop_y1: crop_y2, crop_x1: crop_x2]
                    cv2.imwrite(os.path.join(dst_images_dir, f'{img_name}_{crop_cnt}.png'), img_crop)
                    print(f'{img_name}_{crop_cnt}.png')
                    dst_annot.bbox_map[f'{img_name}_{crop_cnt}'] = new_bboxes
                    crop_cnt += 1
        
        # cvml.write_coco(dst_annot, dst_annotation_path, '.png')
        os.makedirs(os.path.join(dst_dir, split, 'labels'), exist_ok=True)
        cvml.write_yolo_seg(dst_annot, os.path.join(dst_dir, split, 'labels'))
        

def create_splitted_dataset(src_dir: str, dst_dir: str, block_volume: int):
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for sample_name in ['train']:
        sample_path = os.path.join(src_dir, sample_name)
        
        with zipfile.ZipFile(f"{sample_path}.zip", mode="w") as archive:
            directory = Path(sample_path)
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))

        split = Split(f"{sample_path}.zip", dst_dir)
        split.bysize(block_volume)

if __name__ == '__main__':
    main()