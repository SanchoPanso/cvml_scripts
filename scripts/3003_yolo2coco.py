import cvml
import os
import sys




labels_dir = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023\valid_gray\labels'
coco_path = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023\valid_gray\data.json'

annot = cvml.read_yolo(labels_dir, (640, 640), ['tube'])
cvml.write_coco(annot, coco_path, '.png')


# def read_yolo_labels(path: str, img_size: tuple) -> List[BoundingBox]:

#     width, height = img_size
#     image_name = os.path.splitext(os.path.split(path)[-1])[0]
#     with open(path, 'r', encoding='utf-8') as f:
#         rows = f.read().split('\n')
#     bboxes = []
#     segmentation = []
#     for row in rows:
#         if row == '':
#             continue
#         row_data = list(map(float, row.split(' ')))
#         if len(row_data) == 5:
#             cls_id, xc, yc, w, h = row_data
#             # x = xc - w / 2
#             # y = yc - h / 2
#             cls_conf = 1.0
#         elif len(row_data) == 6:
#             cls_id, xc, yc, w, h, cls_conf = row_data
#             # x = xc - w / 2
#             # y = yc - h / 2
#         else:
#             cls_id = row_data[0]
#             cls_conf = 1.0
#             x_coords = row_data[1::2]
#             y_coords = row_data[2::2]
            
#             x1 = min(x_coords)
#             x2 = max(x_coords)
#             y1 = min(y_coords)
#             y2 = max(y_coords)
            
#             xc = (x2 - x1) / 2
#             yc = (y2 - y1) / 2
#             w = (x2 + x1) / 2
#             h = (y2 + y1) / 2
            
#             x_coords = map(lambda x: x * width, x_coords)
#             y_coords = map(lambda y: y * height, y_coords)
            
#             segmentation = [0] * (len(row_data) - 1)
#             segmentation[0::2] = x_coords
#             segmentation[1::2] = y_coords

#         bbox = BoundingBox(cls_id, xc, yc, w, h, cls_conf, 
#                             type_coordinates=CoordinatesType.Relative,
#                             image_name=image_name,
#                             img_size=img_size,
#                             segmentation=segmentation)
#         bboxes.append(bbox)
#     return bboxes


