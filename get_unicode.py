import json
import random

# ① 한글 11,172자 생성
start, end = 0xAC00, 0xD7A3
all_hangul = [chr(u) for u in range(start, end + 1)]

# ② 랜덤으로 train / valid 분리 (80% / 20%)
random.seed(42)  # 재현성을 위해
random.shuffle(all_hangul)
split_idx = int(len(all_hangul) * 0.8)
train_chars = all_hangul[:split_idx]
valid_chars = all_hangul[split_idx:]

# ③ 유니코드 HEX 변환
train_unis = [hex(ord(ch))[2:].upper() for ch in train_chars]
valid_unis = [hex(ord(ch))[2:].upper() for ch in valid_chars]

# ④ JSON 저장
with open("train_unis.json", "w", encoding="utf-8") as f:
    json.dump(train_unis, f, ensure_ascii=False, indent=2)

with open("val_unis.json", "w", encoding="utf-8") as f:
    json.dump(valid_unis, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_unis)} 글자, Valid: {len(valid_unis)} 글자 저장 완료")
