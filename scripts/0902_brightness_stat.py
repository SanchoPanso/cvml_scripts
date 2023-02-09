import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cvml

def save_comet_stats():
    raw_datasets_dir = '/home/student2/datasets/raw'

    raw_dirs = set()
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*cvs1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*csv1*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*number*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '*comet*')))
    raw_dirs |= set(glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир')))

    raw_dirs = list(raw_dirs)
    raw_dirs.sort()

    mean_colors = []
    median_colors = []
    min_colors = []
    max_colors = []

    for dataset_dir in raw_dirs:
        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        annotation = cvml.read_coco(annotation_path)
        
        for image_name in annotation.bbox_map:
            print(dataset_dir + ':' + image_name)
            img = cv2.imread(os.path.join(dataset_dir, 'images', image_name + '.png'))
            if img is None:
                continue
            bboxes = annotation.bbox_map[image_name]
            
            for bbox in bboxes:
                cls_id = bbox.get_class_id()
                if annotation.classes[cls_id] != 'comet':
                    continue
                
                x, y, w, h = map(int, bbox.get_coordinates())
                crop = img[y: y + h, x: x + w]
                
                min_colors.append(crop.min())
                max_colors.append(crop.max())
                mean_colors.append(crop.mean())
                median_colors.append(np.median(crop))

    df = pd.DataFrame(
        {
            'min_color': min_colors,
            'max_color': max_colors,
            'mean_color': mean_colors,
            'median_color': median_colors,
        }
    )

    print(df)
    df.to_csv('cvs1_color_stats.csv')


def show_comet_stats():
    df = pd.read_csv('cvs1_color_stats.csv')
    min_color = df['min_color'].values
    max_color = df['max_color'].values
    
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(min_color, bins=10)
    axs[1].hist(max_color, bins=10)
    
    plt.show()
    
    
            
if __name__ == '__main__':
    #save_comet_stats()
    show_comet_stats()

