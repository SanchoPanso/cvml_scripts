import json
import cv2
import os
import cvml
import shutil


DATASET_DIR = r'C:\Users\HP\Downloads\isaid\isaid'


def fix_train_json(dataset_dir: str):

    data = None
    with open(os.path.join(dataset_dir, 'train', 'iSAID_train.json'), 'r') as f:
        data = json.load(f)

    images_data = data['images']
    print(images_data[0])

    for img_data in images_data:
        file_name = img_data['file_name']
        
        img = None
        img_path = os.path.join(dataset_dir, 'train', 'images', file_name)
        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        img_data['height'] = height
        img_data['width'] = width
        print(img_path)

    print(data['images'])   

    with open(os.path.join(dataset_dir, 'train', 'train.json'), 'w') as f:
        json.dump(data, f)


def create_small_dataset(src_dir, dst_dir):

    for split in ['valid']:
        src_images_dir = os.path.join(src_dir, split, 'images')
        dst_images_dir = os.path.join(dst_dir, split, 'images')
        
        src_coco_path = os.path.join(src_dir, split, 'annotations', f'{split}.json')
        dst_annotations_path = os.path.join(dst_dir, split, 'annotations')
        dst_coco_path = os.path.join(dst_annotations_path, f'{split}.json')
        dst_labels_path = os.path.join(dst_dir, split, 'labels')
        
        os.makedirs(dst_annotations_path, exist_ok=True)
        os.makedirs(dst_labels_path, exist_ok=True)
        os.makedirs(dst_images_dir, exist_ok=True)
        
        src_annot = cvml.read_coco(src_coco_path)
        src_bbox_map = src_annot.bbox_map
        
        names = list(src_bbox_map.keys())
        names.sort()
        names = names[:int(len(names) * 0.1)]
        
        print(split, names)
        
        dst_bbox_map = {name: src_bbox_map[name] for name in names}
        dst_annot = cvml.Annotation(src_annot.classes, dst_bbox_map)
        
        cvml.write_yolo_seg(dst_annot, dst_labels_path)
        cvml.write_coco(dst_annot, dst_coco_path)
        
        for name in names:
            shutil.copy(os.path.join(src_images_dir, name + '.png'), 
                        os.path.join(dst_images_dir, name + '.png'))


def create_cropped(src_dir, dst_dir):
    
    for split in ['train', 'valid']:
    
        src_images_dir = os.path.join(src_dir, split, 'images')
        dst_images_dir = os.path.join(dst_dir, split, 'images')
        
        src_coco_path = os.path.join(src_dir, split, 'annotations', f'{split}.json')
        dst_annotations_path = os.path.join(dst_dir, split, 'annotations')
        dst_coco_path = os.path.join(dst_annotations_path, f'{split}.json')
        dst_labels_path = os.path.join(dst_dir, split, 'labels')
        
        os.makedirs(dst_annotations_path, exist_ok=True)
        os.makedirs(dst_labels_path, exist_ok=True)
        os.makedirs(dst_images_dir, exist_ok=True)
        
        src_annot = cvml.read_coco(src_coco_path)
        
        


if __name__ == '__main__':
    #fix_train_json(DATASET_DIR)
    create_small_dataset(r'C:\Users\HP\Downloads\isaid\isaid', r'C:\Users\HP\Downloads\isaid\isaid_small')