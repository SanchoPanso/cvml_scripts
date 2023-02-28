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
from filesplit.split import Split

import cvml
from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.dataset.image_transforming import convert_to_mixed

import zipfile


def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*false*')))

    raw_dirs = list(raw_dirs)
    raw_dirs.sort()
    
    save_dir = '/home/student2/datasets/prepared/tmk_cvs1_false_yolo_640px_23022023'

    create_compressed_samples = True

    cls_names = ['comet', 'joint', 'number']
    split_proportions = {'valid': 1.0}

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


if __name__ == '__main__':
    main()