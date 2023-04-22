import os
import shutil


percentage = 0.1
orig_dataset_dir = '/content/isaid'
small_dataset_dir = '/content/isaid_small'


os.mkdirs(os.path.join(small_dataset_dir, 'train', 'images'), exist_ok=True)
os.mkdirs(os.path.join(small_dataset_dir, 'train', 'labels'), exist_ok=True)
os.mkdirs(os.path.join(small_dataset_dir, 'valid', 'images'), exist_ok=True)
os.mkdirs(os.path.join(small_dataset_dir, 'valid', 'labels'), exist_ok=True)


for split in ['train', 'valid']:
    for data_type in ['images', 'labels']:
        src_files = os.listdir(os.path.join(orig_dataset_dir, split, data_type))
        src_files.sort()
        src_files = src_files[:int(len(src_files) * percentage)]

        for f in src_files:
            src_path = os.path.join(orig_dataset_dir, split, data_type, f)
            dst_path = os.path.join(small_dataset_dir, split, data_type, f)
            shutil.copy(src_path, dst_path)       
            

