import os
import sys
import glob
import cv2
import numpy as np
import torch
import logging
from typing import Callable, List
import argparse
from pathlib import Path
import zipfile
from filesplit.split import Split
from ultralytics import YOLO

import cvml
from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.dataset.image_transforming import normalize_min_max, SPEstimatorNumpy


save_dir = '/mnt/data/tmk_datasets/prepared/tmk_cvs3_yolo_640px_11042023'
    
raw_datasets_dir = '/mnt/data/tmk_datasets/raw'
raw_dirs = set()
raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'CVS3/*'))) 
raw_dirs -= set(glob.glob(os.path.join(raw_datasets_dir, 'CVS3/SVC3_defects TMK6'))) 
raw_dirs = list(raw_dirs)
raw_dirs.sort()

cls_names = ['other', 'tube', 'sink', 'riska', 'pseudo']
split_proportions = {'train': 0.8, 'valid': 0.2}

tube_weights = '/home/student2/Downloads/train4/weights/best.pt'


def main():
    install_dataset(raw_dirs, save_dir, cls_names, split_proportions)
    create_masks(os.path.join(save_dir, 'train', 'images'), 
                 os.path.join(save_dir, 'train', 'masks'), 
                 tube_weights)
    compress_dataset(save_dir, split_proportions)
    


def install_dataset(raw_dirs: list, save_dir: str, cls_names: list, split_proportions: dict):
    cvml_logger = logging.getLogger('cvml')

    # Create handlers
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_handler.setFormatter(s_format)

    # Add handlers to the logger
    cvml_logger.addHandler(s_handler)


    final_dataset = cvml.DetectionDataset()

    for dataset_dir in raw_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        preprocess_fn = lambda x: cv2.resize(convert_to_mixed(x), (640, 640))
        image_sources = convert_paths_to_single_sources(paths=image_files,
                                                        preprocess_fn=preprocess_fn)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        renamer = lambda x: os.path.split(dataset_dir)[-1] + '_' + x

        annotation_data = cvml.read_coco(annotation_path)
        annotation_data = cvml.change_classes_by_new_classes(annotation_data, cls_names)

        dataset = cvml.DetectionDataset(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset


    final_dataset.split_by_proportions(split_proportions)
    final_dataset.install(
        save_dir, 
        image_ext='.png',
        install_images=True, 
        install_labels=True, 
        install_annotations=True, 
        install_description=True
    )


def create_masks(src_dir, dst_dir, weights):
    img_files = os.listdir(src_dir)
    model = YOLO(weights)
    os.makedirs(dst_dir, exist_ok=True)

    for file in img_files:
        img = cv2.imdecode(np.fromfile(os.path.join(src_dir, file), 'uint8'), cv2.IMREAD_COLOR)
        gray = img[:, :, 2]
        gray = cv2.merge([gray] * 3)
        results = model(gray, stream=True)
        
        img_mask = np.zeros(img.shape[:2], dtype='uint8')
        
        for result in results:
            masks = result.masks
        
        if masks is None:
            continue
        
        data = masks.data
        data = data.cpu().numpy()
        data *= 255
        data = data.astype('uint8')
        
        for obj_data in data:
            obj_data = cv2.resize(obj_data, img_mask.shape[1::-1])
            img_mask += obj_data
        
        cv2.imwrite(os.path.join(dst_dir, file), img_mask)
        print(file)


def compress_dataset(save_dir, split_proportions):
    for sample_name in split_proportions:
        sample_path = os.path.join(save_dir, sample_name)
        # os.system(f" cd {save_dir}; zip -r {sample_name}.zip {sample_name}/*")

        with zipfile.ZipFile(f"{sample_path}.zip", mode="w") as archive:
            directory = Path(sample_path)
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))

        # os.system(f"split {sample_path}.zip {sample_path}.zip.part_ -b 999MB")

        split = Split(f"{sample_path}.zip", save_dir)
        split.bysize(999*1024*1024) # 999MB
    print("Compressed")




def quadratic_lightening(img: np.ndarray, coef: float = 2.35):
    lut = np.zeros((256,), dtype='uint8')
    for i in range(256):
        lut_i = i + coef * (i - i**2 / 255)
        lut[i] = np.int8(np.clip(lut_i, 0, 255))

    if len(img.shape) == 2: # Grayscale case
        img = cv2.LUT(img, lut)
        
    else:   # RGB case
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        hsv = cv2.split(img)
        hsv = (hsv[0], hsv[1], cv2.LUT(hsv[2], lut))   
             
        img = cv2.merge(hsv)                                 
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)     
                
    return img


def convert_to_mixed(orig_img: np.ndarray, estimator: SPEstimatorNumpy = None) -> np.ndarray:

    height, width = orig_img.shape[0:2]
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    #in_data = torch.from_numpy(img).float() #torch.frombuffer(orig_img.data, dtype=torch.uint8, count=img.size).float().detach_().reshape(height, width)
    
    estimator = estimator or SPEstimatorNumpy()
    rho, phi = estimator.getAzimuthAndPolarization(img)
    
    normalized_rho = normalize_min_max(rho)
    normalized_phi = normalize_min_max(phi)

    rho_img = (normalized_rho * 255).astype('uint8')
    phi_img = (normalized_phi * 255).astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray_img = quadratic_lightening(img, 2.35)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    img = cv2.merge([phi_img, rho_img, gray_img])
    return img


if __name__ == '__main__':
    main()