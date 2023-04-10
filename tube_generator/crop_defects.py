import os
import glob
import sys
import cv2
import numpy as np
import torch

import cvml
from cvml.annotation.bounding_box import BoundingBox
from cvml.dataset.image_transforming import SPEstimatorNumpy


def main():
    
    # Dir with tmk datasets for extracting bbox, mask
    tmk_dirs_path = '/mnt/data/tmk_datasets/raw/CVS3'
    save_dir = '/mnt/data/tmk_datasets/crops/1004_defect_crops'
    
    defect_names = ['sink', 'riska']                            # ids for cropping
    crop_counter = {id: 1 for id in defect_names} # counter for naming

    # list of path to tmk dirs
    tmk_dirs = set()
    tmk_dirs |= set(glob.glob(os.path.join(tmk_dirs_path, '*')))


    for tmk_dir in tmk_dirs:
        tmk_dir_path = os.path.join(tmk_dirs_path, tmk_dir)

        crop_counter = crop_defects(tmk_dir_path,
                                    defect_names,
                                    crop_counter,
                                    save_dir)


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



def crop_defects(tmk_dir_path: str,
                 defect_names: list,
                 crop_counters: dict,
                 save_dir: str) -> dict:

    os.makedirs(save_dir, exist_ok=True)

    tmk_dirs_path = os.path.dirname(tmk_dir_path)
    tmk_dir = os.path.split(tmk_dir_path)[-1]

    all_images_paths = glob.glob(os.path.join(tmk_dirs_path, tmk_dir, 'images', '*'))
    color_mask_paths = glob.glob(os.path.join(tmk_dirs_path, tmk_dir, 'images', '*color_mask*'))
    image_paths = list(set(all_images_paths) - set(color_mask_paths))
    annotation_path = os.path.join(tmk_dirs_path, tmk_dir, 'annotations', 'instances_default.json')

    annotation_data = cvml.read_coco(annotation_path)

    # Run through every image in tmk_dir
    for img_path in image_paths:

        # Checking img name in annotation data
        img_name, ext = os.path.splitext(os.path.split(img_path)[-1])
        if img_name not in annotation_data.bbox_map.keys():
            print(img_name, 'is absent')
            continue
        
        # Checking color mask in common list of masks
        color_mask_path = os.path.join(tmk_dir_path, 'images', img_name + '_color_mask.png')
        if not os.path.exists(color_mask_path):
            print(img_name, 'color mask is absent')
            continue

        # get preprocessed image
        img = cv2.imread(img_path)
        mixed_img = convert_to_mixed(img)

        # get binary mask
        mask_img = cv2.imread(color_mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask_img, 1, 1, cv2.THRESH_BINARY)

        # Run through each bbox in the image
        for bbox in annotation_data.bbox_map[img_name]:
            
            # Check class id
            class_id = bbox.get_class_id()
            if annotation_data.classes[class_id] not in defect_names:
                continue
            
            # Check masked img
            masked_img = get_gradient_img(bbox, mixed_img, mask)
            if masked_img is None:
                continue
            
            # Get appropriate crop counter for the class id and increment 
            crop_counter = crop_counters[annotation_data.classes[class_id]]
            crop_counters[annotation_data.classes[class_id]] += 1

            # Save crop in save_dir
            cls_name = annotation_data.classes[class_id]
            os.makedirs(os.path.join(save_dir, cls_name), exist_ok=True)
            np.save(os.path.join(save_dir, cls_name, f'{crop_counter}.npy'), masked_img)
            print(crop_counter)
    
    return crop_counters


def get_masked_img(bbox: BoundingBox, final_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xc, yc, w, h = bbox.get_relative_bounding_box()
    cls_id = bbox.get_class_id()
    
    height, width = final_img.shape[0:2]

    xc *= width
    yc *= height
    w *= width
    h *= height

    xc, yc, w, h = map(int, [xc, yc, w, h])
    padding_x = w // 4
    padding_y = h // 4

    x1 = max(0, xc - w//2 - padding_x)
    x2 = min(width - 1, xc + w//2 + padding_x)
    y1 = max(0, yc - h//2 - padding_y)
    y2 = min(height - 1, yc + h//2 + padding_y)

    final_img_crop = final_img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    masked_img = cv2.bitwise_and(final_img_crop, final_img_crop, mask=mask_crop)
    return masked_img

def get_gradient_img(bbox: BoundingBox, final_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xc, yc, w, h = bbox.get_relative_bounding_box()
    cls_id = bbox.get_class_id()
    
    height, width = final_img.shape[0:2]

    xc *= width
    yc *= height
    w *= width
    h *= height

    xc, yc, w, h = map(int, [xc, yc, w, h])
    padding_x = w // 4
    padding_y = h // 4

    x1 = max(0, xc - w//2 - padding_x)
    x2 = min(width - 1, xc + w//2 + padding_x)
    y1 = max(0, yc - h//2 - padding_y)
    y2 = min(height - 1, yc + h//2 + padding_y)

    final_img_crop = final_img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    
    final_img_crop = resize_saving_ratio(final_img_crop, (256, 256))
    mask_crop = resize_saving_ratio(mask_crop, (256, 256))


    mask_crop = mask_crop.astype('int16')
    
    blur_img_crop = cv2.GaussianBlur(final_img_crop, (45, 45), 3)
    blur_img_crop = blur_img_crop.astype('int16')

    final_img_crop = final_img_crop.astype('int16') - blur_img_crop

    masked_img = final_img_crop * cv2.merge([mask_crop] * 3)
    masked_img = np.concatenate([masked_img, mask_crop.reshape(mask_crop.shape[0], mask_crop.shape[1], 1)], axis=2)

    if masked_img.max() > 255 or masked_img.min() < -256:
        print('aaa')

    return masked_img


def normalize_min_max(data):
    data_min = data.min()
    data_max = data.max()
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data


def resize_saving_ratio(img: np.ndarray, size: tuple):
    width_coef = size[0] / img.shape[1]
    height_coef = size[1] / img.shape[0]
    
    if height_coef < width_coef:
        resized_img = cv2.resize(img, (int(img.shape[1] * height_coef), size[1]))
    else:
        resized_img = cv2.resize(img, ( size[0], int(img.shape[0] * width_coef)))
    
    res_shape = (size[1], size[0], img.shape[2]) if len(img.shape) == 3 else (size[1], size[0])
    res = np.zeros(res_shape, dtype=img.dtype)
    x1 = (res.shape[1] - resized_img.shape[1]) // 2
    x2 = (res.shape[1] + resized_img.shape[1]) // 2
    y1 = (res.shape[0] - resized_img.shape[0]) // 2
    y2 = (res.shape[0] + resized_img.shape[0]) // 2
    
    res[y1:y2, x1:x2] = resized_img
    return res


if __name__ == '__main__':
    main()







