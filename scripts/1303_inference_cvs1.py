import glob
import cv2
import os
import torch
from cvml.annotation.bounding_box import BBType
from cvml import Annotation
from cvml.annotation.annotation_converting import read_yolo, write_yolo
from cvml.inference.yolov5_detector import Yolov5Detector
from cvml.metrics import DetectionEvaluator


weights_path = r'D:\weights\yolov5\yolov5l_cvs1_05032023.pt'
dataset_path = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023\valid'

images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')
predict_path = os.path.join(dataset_path, 'predict')

# detector = Yolov5Detector(weights_path, 'cuda:0')
# det_annot = Annotation(['comet', 'joint', 'number'], {})
# image_files = glob.glob(os.path.join(images_path, '*'))

# for i, file in enumerate(image_files):
#     img = cv2.imread(file)
#     bb = detector(img)
#     name = os.path.splitext(os.path.split(file)[-1])[0]
#     det_annot.bbox_map[name] = bb
#     print(i, name)

# os.makedirs(predict_path, exist_ok=True)
# write_yolo(det_annot, predict_path)

det_annot = read_yolo(predict_path, (2448, 2048), ['comet', 'joint', 'number'])
gt_annot = read_yolo(labels_path, (2448, 2048), ['comet', 'joint', 'number'])

gt_bboxes = []
for name in gt_annot.bbox_map:
    for bb in gt_annot.bbox_map[name]:
        bb.set_bb_type(BBType.GroundTruth)
        gt_bboxes.append(bb)

det_bboxes = []
for name in det_annot.bbox_map:
    for bb in det_annot.bbox_map[name]:
        bb.set_bb_type(BBType.Detected)
        det_bboxes.append(bb)


evaluator = DetectionEvaluator()
metrics = evaluator.GetPascalVOCMetrics(gt_bboxes + det_bboxes, 0.1)

#print(metrics[0]['AP'])

mAP = metrics[0]['AP']
TP = metrics[0]['total TP']
FP = metrics[0]['total FP']
GT = metrics[0]['total positives']
FN = GT - TP

P = TP / (TP + FP)
R = TP / (TP + FN)


print('{0:.3f}'.format(P))
print('{0:.3f}'.format(R))
print('{0:.3f}'.format(mAP))


