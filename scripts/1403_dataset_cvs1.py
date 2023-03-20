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

import cvml
from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.dataset.image_transforming import normalize_min_max, SPEstimatorNumpy



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




def main():
    save_dir = '/home/student2/datasets/prepared/tmk_cvs1_yolo_640px_14032023'
    base_dataset = '/home/student2/datasets/prepared/tmk_cvs1_yolo_640px_05032023' 
    
    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'number_March1'))) 
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'TMK_number_march_Marina_1'))) 
    raw_dirs = list(raw_dirs)
    raw_dirs.sort()

    create_compressed_samples = True

    cls_names = ['comet', 'joint', 'number']
    split_proportions = {'train': 0.8, 'valid': 0.2}

    cvml_logger = logging.getLogger('cvml')

    # Create handlers
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_handler.setFormatter(s_format)

    # Add handlers to the logger
    cvml_logger.addHandler(s_handler)
    


    image_sources = convert_paths_to_single_sources(paths=glob.glob(os.path.join(base_dataset, 'train', 'images', '*')),
                                                    preprocess_fn=lambda x: x)
    annotation_path = os.path.join(base_dataset, 'train', 'labels')
    annotation_data = cvml.read_yolo(annotation_path, (640, 640), cls_names)
    train_anchor_dataset = cvml.DetectionDataset(image_sources, annotation_data)

    image_sources = convert_paths_to_single_sources(paths=glob.glob(os.path.join(base_dataset, 'valid', 'images', '*')),
                                                    preprocess_fn=lambda x: x)
    annotation_path = os.path.join(base_dataset, 'valid', 'labels')
    annotation_data = cvml.read_yolo(annotation_path, (640, 640), cls_names)
    valid_anchor_dataset = cvml.DetectionDataset(image_sources, annotation_data)



    final_dataset = cvml.DetectionDataset()
    final_dataset += train_anchor_dataset
    final_dataset += valid_anchor_dataset

    final_dataset.samples = {
        'train': [i for i in range(len(train_anchor_dataset))],
        'valid': [i for i in range(len(train_anchor_dataset), len(final_dataset))],
    }

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

        final_dataset = final_dataset.add_with_proportion(dataset, {'train': 0.8, 'valid': 0.2})

    # final_dataset.split_by_proportions(split_proportions)
    final_dataset.install(
        save_dir, 
        image_ext='.png',
        install_images=True, 
        install_labels=True, 
        install_annotations=True, 
        install_description=True
    )
    
    if create_compressed_samples:
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


if __name__ == '__main__':
    main()