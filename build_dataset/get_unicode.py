import json
import os
import os.path as osp

train_data_dir = "/home/dev/Project/VQ-Font/datasets/train_font_image"
valid_data_dir = "/home/dev/Project/VQ-Font/datasets/valid_font_image"

train_names = []
valid_names = []
for folderName in os.listdir(train_data_dir):
    if folderName == "reference_images": continue
    
    for fileName in os.listdir(osp.join(train_data_dir, folderName)):
        train_names.append(fileName.split('.')[0])

for folderName in os.listdir(valid_data_dir):
    for fileName in os.listdir(osp.join(valid_data_dir, folderName)):
        valid_names.append(fileName.split('.')[0])

train_chars = list(set(train_names))
valid_chars = list(set(valid_names))

# ③ 유니코드 HEX 변환
train_unis = [hex(ord(ch))[2:].upper() for ch in train_chars]
valid_unis = [hex(ord(ch))[2:].upper() for ch in valid_chars]

# ④ JSON 저장
with open("./build_dataset/train_unis_v2.json", "w", encoding="utf-8") as f:
    json.dump(train_unis, f, ensure_ascii=False, indent=2)

with open("./build_dataset/val_unis_v2.json", "w", encoding="utf-8") as f:
    json.dump(valid_unis, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_unis)} 글자, Valid: {len(valid_unis)} 글자 저장 완료")
