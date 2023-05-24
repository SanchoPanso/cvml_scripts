import os
import shutil


percentage = 0.1
orig_dataset_dir = r'C:\Users\HP\Downloads\isaid\isaid'
small_dataset_dir = r'C:\Users\HP\Downloads\isaid\isaid_small'


os.makedirs(os.path.join(small_dataset_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(small_dataset_dir, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(small_dataset_dir, 'valid', 'images'), exist_ok=True)
os.makedirs(os.path.join(small_dataset_dir, 'valid', 'labels'), exist_ok=True)


for split in ['train', 'valid']:
    for data_type in ['images', 'labels']:
        src_files = os.listdir(os.path.join(orig_dataset_dir, split, data_type))
        src_files.sort()
        src_files = src_files[:int(len(src_files) * percentage)]

        for f in src_files:
            src_path = os.path.join(orig_dataset_dir, split, data_type, f)
            dst_path = os.path.join(small_dataset_dir, split, data_type, f)
            shutil.copy(src_path, dst_path)       
            
            
classes = ["storage_tank",
           "Large_Vehicle",
           "Small_Vehicle",
           "ship",
           "Harbor",
           "baseball_diamond",
           "Ground_Track_Field",
           "Soccer_ball_field",
           "Swimming_pool",
           "Roundabout",
           "tennis_court",
           "basketball_court",
           "plane",
           "Helicopter",
           "Bridge"]

description = f"train: /content/isaid_small/train/images\n"\
              f"val: /content/isaid_small/valid/images\n"\
              f"nc: {len(classes)}\n"\
              f"names: {str(classes)}\n"
              
print(description)
with open(os.path.join(small_dataset_dir, 'data.yaml'), 'w') as f:
    f.write(description)
