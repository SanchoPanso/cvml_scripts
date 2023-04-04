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


def main():
    save_dir = '/home/student2/datasets/prepared/tmk_tube_yolo_640px_04042023'
    
    raw_datasets_dir = '/home/student2/datasets/raw/other'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_valid_0503')))  
    raw_dirs = list(raw_dirs)
    raw_dirs.sort()

    create_compressed_samples = True

    cls_names = ['tube']
    split_proportions = {'train': 0.9, 'valid': 0.1}

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

        preprocess_fn = lambda x: cv2.split(x)[2]
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
    
    if create_compressed_samples:
        for sample_name in split_proportions:
            sample_path = os.path.join(save_dir, sample_name)
            
            with zipfile.ZipFile(f"{sample_path}.zip", mode="w") as archive:
                directory = Path(sample_path)
                for file_path in directory.rglob("*"):
                    archive.write(file_path, arcname=file_path.relative_to(directory))

            
            split = Split(f"{sample_path}.zip", save_dir)
            split.bysize(999*1024*1024) # 999MB
        print("Compressed")


if __name__ == '__main__':
    main()