import os
import cv2
import glob
import numpy as np
import cvml2
import json

def main():
    src_annot = r'C:\Users\HP\Downloads\Telegram Desktop\готовая разметка\3768.json'
    src_image_dir = r'D:\datasets\geo_ai\geoai_aerial\geoai_aerial_3768'
    
    dst_annot = r'D:\datasets\geo_ai\geoai_aerials\geoai_aerial_16062023_3768\labels'
    dst_image_dir = r'D:\datasets\geo_ai\geoai_aerials\geoai_aerial_16062023_3768\images'
    size = (1000, 1000)
    isthing = [False, True, True, True, True, True, True, False, False, False]
    filtering_cls_id = 8
    
    # crop_images(src_image_dir, dst_image_dir, size)
    # annot = cvml2.read_coco(src_annot)
    # annot = make_cropped_annotation(annot, size, isthing)
    # with open('intermediate.json', 'w') as f:
    #     json.dump(annot, f)
    
    with open('intermediate.json', 'r') as f:
        annot = json.load(f)
    filter_mask(annot, dst_image_dir, filtering_cls_id)
    # cvml2.write_yolo_iseg(annot, dst_annot)
    
    image_sources = cvml2.convert_paths_to_single_sources(glob.glob(dst_image_dir + r'\*'), lambda x: x)
    dataset = cvml2.ISDataset(image_sources, annot)
    dataset.split_by_proportions({'train': 0.8, 'valid': 0.2})
    dataset.install(
        dataset_path=r'D:\datasets\geo_ai\geoai_aerial\geoai_aerial_18062023',
        dataset_name='geoai_aerial',
        install_images=True,
        install_labels=True,
        install_description=True,
    )    


def crop_images(src_dir, dst_dir, size):
    os.makedirs(dst_dir, exist_ok=True)
    crop_h, crop_w = size
    for img_fn in os.listdir(src_dir):
        img = cv2.imread(os.path.join(src_dir, img_fn))
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
                
                crop = img[crop_y1: crop_y2, crop_x1: crop_x2]
                
                name, ext = os.path.splitext(img_fn)
                crop_fn = f'{name}_{crop_cnt}{ext}'
                cv2.imwrite(os.path.join(dst_dir, crop_fn), crop)
                crop_cnt += 1


def make_cropped_annotation(annotation: dict, size: tuple, isthing: list) -> dict:
    crop_h, crop_w = size
    classes = annotation['categories']

    new_annotation = {
        'categories': annotation['categories'],
        'images': {},
    }
    
    for img_name in annotation['images']:
        
        img_h = annotation['images'][img_name]['height']
        img_w = annotation['images'][img_name]['width']
        bboxes = annotation['images'][img_name]['annotations']
        
        num_of_rows = img_h // crop_h if img_h % crop_h == 0 else img_h // crop_h + 1
        num_of_cols = img_w // crop_w if img_w % crop_w == 0 else img_w // crop_w + 1
        crop_cnt = 1
        
        # Create semantic masks
        semantic_masks = {}
        for i in range(len(classes)):
            if not isthing[i]:
                mask = np.zeros((img_h, img_w), dtype='uint8')
                semantic_masks[i] = mask
        
        # Fill semantic masks with stuff objects
        for bbox in bboxes:            
            segmentation = bbox['segmentation']
            cls_id = bbox['category_id']
            
            if isthing[cls_id]:
                continue
            
            for s in segmentation:
                pts = np.array(s, dtype='int32').reshape((-1, 1, 2))
                cv2.fillPoly(semantic_masks[cls_id], [pts], 255)

        # Create crop annotation
        for row in range(num_of_rows):
            for col in range(num_of_cols):
                
                crop_name = f'{img_name}_{crop_cnt}'
                print(crop_name)
                new_annotation['images'][crop_name] = {
                    'width': crop_w,
                    'height': crop_h,
                    'annotations': [],
                }
                
                new_bboxes = []
                
                crop_x1 = crop_w * col
                crop_y1 = crop_h * row

                crop_x2 = min(crop_x1 + crop_w, img_w)
                crop_y2 = min(crop_y1 + crop_h, img_h)
                
                # Create bboxes for thing objects 
                for bbox in bboxes:
                    
                    segmentation = bbox['segmentation']
                    cls_id = bbox['category_id']
                    
                    if not isthing[cls_id]:
                        continue
                    
                    thing_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype='uint8')
                    for s in segmentation:
                        pts = np.array(s, dtype='int32').reshape((-1, 1, 2))
                        pts[:, 0, 0] -= crop_x1
                        pts[:, 0, 1] -= crop_y1
                        cv2.fillPoly(thing_mask, [pts], (255, 255, 255))
                    
                    contours, hierarchy = cv2.findContours(thing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) == 0:
                        continue
                    
                    c = contours[0]
                    
                    if cv2.contourArea(c) < 10:
                        continue
                    
                    x1 = int(c[:, 0, 0].min())
                    y1 = int(c[:, 0, 1].min())
                    x2 = int(c[:, 0, 0].max())
                    y2 = int(c[:, 0, 1].max())
                    w = x2 - x1
                    h = y2 - y1
                    
                    new_segmentation = c.reshape((1, -1)).tolist()
                    
                    new_bbox = {
                        'category_id': cls_id,
                        'bbox': [x1, y1, w, h],
                        'bbox_mode': 'xywh',
                        'segmentation': new_segmentation,
                    }
                    new_bboxes.append(new_bbox)
                
                # Create bboxes for stuff objects
                for cls_id in range(len(classes)):
                    
                    if isthing[cls_id]:
                        continue
                    
                    semantic_crop = semantic_masks[cls_id][crop_y1: crop_y2, crop_x1: crop_x2]         
                    contours, hierarchy = cv2.findContours(semantic_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                    for c in contours:
                        
                        if cv2.contourArea(c) < 10:
                            continue
                        
                        x1 = int(c[:, 0, 0].min())
                        y1 = int(c[:, 0, 1].min())
                        x2 = int(c[:, 0, 0].max())
                        y2 = int(c[:, 0, 1].max())
                        w = x2 - x1
                        h = y2 - y1
                        
                        new_segmentation = c.reshape((1, -1)).tolist()
                        
                        new_bbox = {
                            'category_id': cls_id,
                            'bbox': [x1, y1, w, h],
                            'bbox_mode': 'xywh',
                            'segmentation': new_segmentation,
                        }
                        new_bboxes.append(new_bbox)
                
                
                new_annotation['images'][crop_name]['annotations'] = new_bboxes
                crop_cnt += 1
    
    return new_annotation


def filter_mask(annotation: dict, image_dir: str, filtering_cls_id: int):
    
    for img_name in annotation['images']:
        
        image_paths = glob.glob(os.path.join(image_dir, img_name + '*'))
        if len(image_paths) == 0:
            continue
        img = cv2.imread(image_paths[0])[:, :, 1]#, cv2.IMREAD_GRAYSCALE)
        
        img_h = annotation['images'][img_name]['height']
        img_w = annotation['images'][img_name]['width']
        bboxes = annotation['images'][img_name]['annotations']
        
        mask = np.zeros((img_h, img_w), dtype='uint8')
        new_bboxes = []
        
        for bbox in bboxes:
            segmentation = bbox['segmentation']
            cls_id = bbox['category_id']
            
            if cls_id != filtering_cls_id:
                new_bboxes.append(bbox)
                continue
            
            for s in segmentation:
                pts = np.array(s, dtype='int32').reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
        
        if mask.sum() == 0:
            annotation['images'][img_name]['annotations'] = new_bboxes
            continue
        
        thresh = int((img * mask).sum() / mask.sum() * 0.8)
        ret, f_mask = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY_INV)
        
        res_mask = mask * f_mask
        contours, hierarchy = cv2.findContours(res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            
            if cv2.contourArea(c) < 50:
                continue
            
            x1 = int(c[:, 0, 0].min())
            y1 = int(c[:, 0, 1].min())
            x2 = int(c[:, 0, 0].max())
            y2 = int(c[:, 0, 1].max())
            w = x2 - x1
            h = y2 - y1
            
            new_segmentation = c.reshape((1, -1)).tolist()

            new_bbox = {
                'category_id': filtering_cls_id,
                'bbox': [x1, y1, w, h],
                'bbox_mode': 'xywh',
                'segmentation': new_segmentation,
            }
            new_bboxes.append(new_bbox)
        annotation['images'][img_name]['annotations'] = new_bboxes
    
    return annotation
        

if __name__ == '__main__':
    main()
