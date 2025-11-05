import json
import os
import os.path as osp

train_data_dir = "/home/dev/VQ-Font/datasets/train_font_image"
valid_data_dir = "/home/dev/VQ-Font/datasets/valid_font_image"

train_names = []
valid_names = []
for folderName in os.listdir(train_data_dir):
    for fileName in os.listdir(osp.join(train_data_dir, folderName)):
        if ('2' in fileName.split('.')[0]) or ('3' in fileName.split('.')[0]): continue
        train_names.append(fileName.split('.')[0])

for folderName in os.listdir(valid_data_dir):
    for fileName in os.listdir(osp.join(valid_data_dir, folderName)):
        valid_names.append(fileName.split('.')[0])

train_chars = list(set(train_names))
valid_chars = list(set(valid_names))

# # ① 한글 11,172자 생성
# start, end = 0xAC00, 0xD7A3
# all_hangul = [chr(u) for u in range(start, end + 1)]

# # ② 랜덤으로 train / valid 분리 (80% / 20%)
# random.seed(42)  # 재현성을 위해
# random.shuffle(all_hangul)
# split_idx = int(len(all_hangul) * 0.8)
# train_chars = all_hangul[:split_idx]
# valid_chars = all_hangul[split_idx:]


# ③ 유니코드 HEX 변환
train_unis = [hex(ord(ch))[2:].upper() for ch in train_chars]
valid_unis = [hex(ord(ch))[2:].upper() for ch in valid_chars]

# ④ JSON 저장
with open("./build_dataset/train_unis.json", "w", encoding="utf-8") as f:
    json.dump(train_unis, f, ensure_ascii=False, indent=2)

with open("./build_dataset/val_unis.json", "w", encoding="utf-8") as f:
    json.dump(valid_unis, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_unis)} 글자, Valid: {len(valid_unis)} 글자 저장 완료")
