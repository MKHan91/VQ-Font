import os.path as osp
import os
from glob import glob

train_font_image_dir = r"/home/dev/VQ-Font/datasets/train_font_image"
val_font_image_dir = r"/home/dev/VQ-Font/datasets/valid_font_image"

train_save_path = r"/home/dev/VQ-Font/vqgan_data/valid_custom.txt"
with open(train_save_path, "w", encoding='utf-8') as f:
    for folderName in os.listdir(val_font_image_dir):
        image_paths = glob(osp.join(val_font_image_dir, folderName, "*.png"))
        
        for image_path in image_paths:
            f.write(image_path + '\n')