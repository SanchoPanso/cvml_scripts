import os
import sys
import cvml
from cvml.annotation.bounding_box import BoundingBox

sorted_digits_dir = '/home/student2/Downloads/number_december1_cropped_sorted'
orig_annot_path = '/home/student2/datasets/raw/number_december1/annotations/instances_default.json'
result_annot_path = '/home/student2/datasets/raw/number_december1/annotations/digits.json'

annotation = cvml.read_coco(orig_annot_path)
new_annotation = cvml.Annotation([str(i) for i in range(9)])

for image_name in annotation.bbox_map:
    bboxes = annotation.bbox_map[image_name]
    new_annotation.bbox_map[image_name] = []
    crop_cnt = 0

    for bbox in bboxes:
        cls_id = bbox.get_class_id()
        if annotation.classes[cls_id] != 'number':
            continue
        
        crop_name = f'number_december1_{image_name}_{crop_cnt}.png'
        for i in range(9):
            if os.path.exists(os.path.join(sorted_digits_dir, str(i), crop_name)):
                print(os.path.join(sorted_digits_dir, str(i), crop_name))
                bbox.set_class_id(i)
                new_annotation.bbox_map[image_name].append(bbox)
                crop_cnt += 1
                break
        
cvml.write_coco(new_annotation, result_annot_path, '.png')



