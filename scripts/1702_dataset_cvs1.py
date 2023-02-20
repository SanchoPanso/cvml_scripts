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

import cvml
from cvml.dataset.image_source import convert_paths_to_single_sources
from cvml.dataset.image_transforming import convert_to_mixed



def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*cvs1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*csv1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*number*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*comet*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир')))

    raw_dirs = list(raw_dirs)
    raw_dirs.sort()
    
    save_dir = '/home/student2/datasets/prepared/tmk_cvs1_yolo_640px_17022023'

    create_compressed_samples = True

    cls_names = ['comet', 'joint', 'number', 'tube', 'highlight']
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

    
    final_dataset = cvml.DetectionDataset()
    
    for dataset_dir in raw_dirs:
        
        image_dir = os.path.join(dataset_dir, 'images')
        all_files = glob.glob(os.path.join(image_dir, '*'))
        color_masks_files = glob.glob(os.path.join(image_dir, '*color_mask*'))
        image_files = list(set(all_files) - set(color_masks_files))
        print(len(image_files), dataset_dir)

        preprocess_fn = lambda x: cv2.resize(convert_to_mixed(x), (640, 535))
        image_sources = convert_paths_to_single_sources(paths=image_files,
                                                        preprocess_fn=preprocess_fn)

        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_with_highlights.json')
        renamer = lambda x: os.path.split(dataset_dir)[-1] + '_' + x

        annotation_data = cvml.read_coco(annotation_path)
        classes = cls_names
        annotation_data = cvml.change_classes_by_new_classes(annotation_data, classes)

        dataset = cvml.DetectionDataset(image_sources, annotation_data)
        dataset.rename(renamer)

        final_dataset += dataset

    sample_proportions = split_proportions
    final_dataset.split_by_proportions(sample_proportions)
    final_dataset.install(
        save_dir, 
        image_ext='.png',
        install_images=True, 
        install_labels=True, 
        install_annotations=True, 
        install_description=True
    )
    
    if create_compressed_samples:
        if os.name == 'posix':
            all_samples = list(sample_proportions.keys())

            for sample_name in all_samples:
                sample_path = os.path.join(save_dir, sample_name)
                os.system(f" cd {save_dir}; zip -r {sample_name}.zip {sample_name}/*")
                os.system(f"split {sample_path}.zip {sample_path}.zip.part_ -b 999MB")
        elif os.name == 'nt':
            os.system("echo Zip is not Implemented")    # TODO
        else:
            pass


if __name__ == '__main__':
    main()