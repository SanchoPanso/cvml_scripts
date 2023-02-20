import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cvml
from cvml.dataset.image_transforming import convert_to_mixed

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

    img_gray_mean = []
    img_rho_mean = []
    img_phi_mean = []
    
    bbox_gray_mean = []
    bbox_gray_median = []
    bbox_gray_min = []
    bbox_gray_max = []
    
    bbox_rho_mean = []

    for dataset_dir in raw_dirs:
        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
        annotation = cvml.read_coco(annotation_path)
        
        for image_name in annotation.bbox_map:
            print(dataset_dir + ':' + image_name)
            img = cv2.imread(os.path.join(dataset_dir, 'images', image_name + '.png'))
            if img is None:
                continue
            mixed_img = convert_to_mixed(img)
            phi, rho, gray = cv2.split(mixed_img)
            
            cur_img_gray_mean = gray.mean()
            cur_img_rho_mean = rho.mean()
            cur_img_phi_mean = phi.mean()
            
            bboxes = annotation.bbox_map[image_name]
            
            for bbox in bboxes:
                cls_id = bbox.get_class_id()
                if annotation.classes[cls_id] != 'comet':
                    continue
                
                x, y, w, h = map(int, bbox.get_coordinates())
                crop = img[y: y + h, x: x + w]
                rho_crop = rho[y: y + h, x: x + w]
                
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                bbox_gray_min.append(crop.min())
                bbox_gray_max.append(crop.max())
                bbox_gray_mean.append(crop.mean())
                bbox_gray_median.append(np.median(crop))
                
                bbox_rho_mean.append(rho_crop.mean())
                
                img_gray_mean.append(cur_img_gray_mean)
                img_rho_mean.append(cur_img_rho_mean)
                img_phi_mean.append(cur_img_phi_mean)

    df = pd.DataFrame(
        {
            'bbox_min_gray': bbox_gray_min,
            'bbox_max_gray': bbox_gray_max,
            'bbox_mean_gray': bbox_gray_mean,
            'bbox_median_gray': bbox_gray_median,
            
            'bbox_rho_mean': bbox_rho_mean,
            
            'img_gray_mean': img_gray_mean,
            'img_rho_mean': img_rho_mean,
            'img_phi_mean': img_phi_mean,
        }
    )

    print(df)
    df.to_csv('cvs1_color_stats.csv')


def save_false_comet_stats():
    
    raw_dirs = [r'C:\Users\HP\Downloads\TMK_false_defects_february\TMK_false_defects_february\TMK_false_defects_february']

    mean_colors = []
    median_colors = []
    min_colors = []
    max_colors = []

    for dataset_dir in raw_dirs:
        annotation_path = os.path.join(dataset_dir, 'predicts', 'labels')
        annotation = cvml.read_yolo(annotation_path, (2448, 2048), ['comet', 'joint', 'number'])
        
        for image_name in annotation.bbox_map:
            print(dataset_dir + ':' + image_name)
            img = cv2.imread(os.path.join(dataset_dir, 'images', image_name + '.png'))
            if img is None:
                continue
            bboxes = annotation.bbox_map[image_name]
            
            for bbox in bboxes:
                cls_id = bbox.get_class_id()
                if annotation.classes[int(cls_id)] != 'comet':
                    continue
                
                x, y, w, h = map(int, bbox.get_coordinates())
                crop = img[y: y + h, x: x + w]
                
                if w == 0 or h == 0:
                    continue
                
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
    df.to_csv('cvs1_false_comet_stats.csv')


def show_comet_stats():
    df = pd.read_csv('cvs1_color_stats.csv')
    min_color = df['min_color'].values
    max_color = df['max_color'].values
    mean_color = df['mean_color'].values
    median_color = df['median_color'].values
    
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0][0].hist(min_color, bins=20)
    axs[0][1].hist(max_color, bins=20)
    axs[1][0].hist(mean_color, bins=20)
    axs[1][1].hist(median_color, bins=20)
    
    axs[0][0].set_title('min color')
    axs[0][1].set_title('max color')
    axs[1][0].set_title('mean color')
    axs[1][1].set_title('median color')
    
    plt.show()
    

def show_false_comet_stats():
    df = pd.read_csv('cvs1_false_comet_stats.csv')
    columns = df.columns
    
    fig, axs = plt.subplots(1, len(columns), sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0][0].hist(min_color, bins=20)
    axs[0][1].hist(max_color, bins=20)
    axs[1][0].hist(mean_color, bins=20)
    axs[1][1].hist(median_color, bins=20)
    
    axs[0][0].set_title('min color')
    axs[0][1].set_title('max color')
    axs[1][0].set_title('mean color')
    axs[1][1].set_title('median color')
    plt.show()
            
if __name__ == '__main__':
    #save_comet_stats()
    show_comet_stats()
    
    #save_false_comet_stats()
    show_false_comet_stats()

