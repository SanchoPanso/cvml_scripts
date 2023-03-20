import os
import sys
import random
import glob
import cvml
import logging
from cvml.dataset.image_source import convert_paths_to_single_sources


dataset_path = '/home/student2/datasets/prepared/tmk_cvs1_yolo_640px_14032023'
save_path = '/home/student2/datasets/prepared/subset_tmk_cvs1_yolo_640px_14032023'
subset_percent = 0.1


cvml_logger = logging.getLogger('cvml')

# Create handlers
s_handler = logging.StreamHandler(sys.stdout)
s_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
s_handler.setFormatter(s_format)

# Add handlers to the logger
cvml_logger.addHandler(s_handler)


train_images_path = os.path.join(dataset_path, 'train', 'images')
train_labels_path = os.path.join(dataset_path, 'train', 'labels')

valid_images_path = os.path.join(dataset_path, 'valid', 'images')
valid_labels_path = os.path.join(dataset_path, 'valid', 'labels')

classes = ['comet', 'joint', 'number']

train_annot = cvml.read_yolo(train_labels_path, (640, 640), classes)
valid_annot = cvml.read_yolo(valid_labels_path, (640, 640), classes)

train_images = convert_paths_to_single_sources(glob.glob(os.path.join(train_images_path, '*')), lambda x: x)
valid_images = convert_paths_to_single_sources(glob.glob(os.path.join(valid_images_path, '*')), lambda x: x)

train_dataset = cvml.DetectionDataset(train_images, train_annot)
valid_dataset = cvml.DetectionDataset(valid_images, valid_annot)


train_subset_ids = [i for i in range(len(train_dataset))]
valid_subset_ids = [i + len(train_dataset) for i in range(len(valid_dataset))]

random.shuffle(train_subset_ids)
random.shuffle(valid_subset_ids)

train_subset_ids = train_subset_ids[:int(subset_percent * len(train_subset_ids))]
valid_subset_ids = valid_subset_ids[:int(subset_percent * len(valid_subset_ids))]


subset = train_dataset + valid_dataset
subset.samples = {
    'train': train_subset_ids,
    'valid': valid_subset_ids,
}

subset.install(
    save_path,
    '.png',
)
