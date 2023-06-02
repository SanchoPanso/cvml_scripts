import os
import numpy as np
import cvml
from cvml.annotation.bounding_box import BoundingBox, BBType, BBFormat

src_path = r'C:\Users\HP\Downloads\09CM_3863.json'
dst_path = r'C:\Users\HP\Downloads\09CM_3863_new.json'

annot = cvml.read_coco(src_path)

for img_name in annot.bbox_map:
    bboxes = annot.bbox_map[img_name]
    if len(bboxes) == 0:
        continue
    
    groups = {}
    for bbox in bboxes:
        id = bbox.get_class_id()
        if id not in groups:
            groups[id] = []
        groups[id].append(bbox)
    
    new_bboxes = []
    for id in groups:
        print(id)
        group = groups[id]
        old_segments = []
        
        img_size = group[0].get_image_size()
        
        for i, bbox in enumerate(group):
            s = bbox.get_segmentation()[0]
            old_segments.append(s)
            if i == 1:
                break
        
        
        first_new_segm = []
        for s in old_segments:
            first_new_segm += s
            
        second_new_segm = []
        for s in old_segments[::-1]:
            second_new_segm += [s[-1], s[-2], s[0], s[1]]
        second_new_segm = second_new_segm[2:-2]
        
        new_segm = first_new_segm + second_new_segm
        
        x_points = new_segm[0::2]
        y_points = new_segm[1::2]
        
        x = min(x_points)
        x2 = max(x_points)
        w = x2 - x
        y = min(y_points)
        y2 = max(y_points)
        h = y2 - y
        
        new_bbox = BoundingBox(id, x, y, w, h, 1, img_name, img_size=img_size, segmentation=new_segm)
        new_bboxes.append(new_bbox)
        break
    
    annot.bbox_map[img_name] = new_bboxes

cvml.write_coco(annot, dst_path)
        
        
        
            
            
        










