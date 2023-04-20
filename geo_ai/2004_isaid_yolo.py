import json
import cv2
import os
import cvml


DATASET_DIR = '/mnt/data/geo_ai_datasets/isaid'


def fix_train_json(dataset_dir: str):

    data = None
    with open(os.path.join(dataset_dir, 'train', 'iSAID_train.json'), 'r') as f:
        data = json.load(f)

    images_data = data['images']
    print(images_data[0])

    for img_data in images_data:
        file_name = img_data['file_name']
        
        img = None
        for part in ['part1', 'part2', 'part3']:
            img_path = os.path.join(dataset_dir, 'train', part, 'images', file_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                break

        height, width = img.shape[0], img.shape[1]
        img_data['height'] = height
        img_data['width'] = width
        print(img_path)

    print(data['images'])   

    with open(os.path.join(dataset_dir, 'train', 'train.json'), 'w') as f:
        json.dump(data, f)



if __name__ == '__main__':
    fix_train_json(DATASET_DIR)