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
import shutil

import cvml
from cvml.annotation.bounding_box import BoundingBox, CoordinatesType, BBType, BBFormat
from cvml.dataset.detection_dataset import DetectionDataset
from cvml.dataset.image_source import ImageSource



def main():

    raw_datasets_dir = '/home/student2/datasets/raw'
    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number1')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number2')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number3')))

    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'number_december1')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'number_december2')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number_october1')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, 'cvs1_number_january_Marina1')))

    save_dir = '/home/student2/datasets/raw/digits_annotation'
    
    for raw_dir in raw_dirs:
        images_dir = os.path.join(raw_dir, 'images')
        annotations_dir = os.path.join(raw_dir, 'annotations')
        annotations_path = os.path.join(annotations_dir, 'digits.json')

        new_images_dir = os.path.join(save_dir, os.path.split(raw_dir)[-1], 'images')
        new_annotations_dir = os.path.join(save_dir, os.path.split(raw_dir)[-1], 'annotations')
        new_annotations_path = os.path.join(new_annotations_dir, 'digits.json')

        os.makedirs(new_annotations_dir, exist_ok=True)
        shutil.copy(annotations_path, new_annotations_path)

        print(raw_dir)
        

if __name__ == '__main__':
    main()