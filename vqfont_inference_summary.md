# VQ-Font 붓글씨 추론 파이프라인 정리

## 1. 현황 요약

| 항목 | 값 |
|------|-----|
| 학습 체크포인트 | `brush_finetune_v2/last.ckpt` (544.4 MB) |
| 학습 step | 41,000 / 50,000 (Iteration finished) |
| 최종 loss | 0.325 (시작 1.033 → 최종 0.325) |
| generator_ema | ✅ 존재 (추론용) |
| reference_images_v2 | 78장 |
| cr_mapping_v2 구성요소 | 78개 (100% 커버) |
| **추론 가능 글자** | **11,172 / 11,172 (100%)** |

---

## 2. 추론 코드 수정 사항

### 2-1. `inference.py` — 11,172자 전체 생성

**변경 전**: 구성요소 3개가 모두 reference에 있는 글자만 추론 대상에 포함  
**변경 후**: `cr_mapping`의 모든 11,172자를 무조건 추론 대상에 포함

```python
# 변경 후 (inference.py getMetaDict 함수)
infer_unis = [chr(int(uni, 16)) for uni in cr_mapping.keys()]
```

### 2-2. `dataset_transformer.py` — 자모 유사도 기반 대체

누락된 구성요소가 있을 경우(현재는 발생하지 않지만 fallback) 랜덤 대체 → **자모 유사도 기반** 대체로 변경.

- 초성·중성·종성 분해 후 일치 개수(0~3)로 유사도 계산
- 보유 글자 중 가장 유사한 글자 선택

### 2-3. `dataset_transformer.py` — 추론 시 Augmentation 비활성화

**변경 전**: 추론 시에도 랜덤 회전/왜곡/블러 적용  
**변경 후**: `ret_targets=False`(추론 모드)이면 augmentation 건너뜀

```python
if self.ret_targets:  # 학습 모드에서만 augmentation
    ...
```

### 2-4. weight 경로 업데이트

```
brush_finetune_v1/last.ckpt → brush_finetune_v2/last.ckpt
```

---

## 3. 추론 실행 방법

```bash
cd /home/dev/Project/VQ-Font
python3 inference.py
```

### 기본 인자값

| 인자 | 값 |
|------|-----|
| `--weight` | `vq_font_results/checkpoints/brush_finetune_v2/last.ckpt` |
| `--content_font` | `datasets/content_font_image/NanumBarunpenR` |
| `--img_path` | `datasets/train_font_image/reference_images_v2` |
| `--saving_root` | `./inference_results/target_style_images` |
| batch_size | 8 (코드 내 설정) |

### 결과 저장 경로

```
./inference_results/target_style_images/reference_images_v2/images/가.png
./inference_results/target_style_images/reference_images_v2/images/각.png
...
(11,172개 PNG 파일)
```

---

## 4. 추론 시 주의사항

| # | 항목 | 설명 |
|---|------|------|
| 1 | VRAM | batch_size=8 기준 약 6~8GB 사용. 부족하면 4로 줄이기 |
| 2 | 소요 시간 | 11,172자 / batch 8 ≈ 1,397 배치. GPU에 따라 10~30분 |
| 3 | 체크포인트 선택 | `generator_ema` 우선 로드 (EMA가 더 안정적) |
| 4 | 중간 체크포인트 | 결과 불만족 시 `040000-brush_finetune_v2.ckpt` 등 시도 |

---

## 5. 학습 검증 결과

### Loss 추이

```
Step  5,000 → 1.033
Step 10,000 → 0.742
Step 15,000 → 0.877 (GAN 변동)
Step 20,000 → 0.504
Step 25,000 → 0.812 (GAN 변동)
Step 30,000 → 0.603
Step 35,000 → 0.525
Step 40,000 → 0.348
Step 41,000 → 0.325 (last)
```

### 가중치 변화 확인

- 5000 step vs last 평균 변화량: 0.0025 ✅
- 가장 많이 변한 레이어: `vqgan.decoder.layers.6.conv.weight` (diff=0.034)
- decoder 레이어 중심으로 학습됨 → 스타일 생성 부분 정상 업데이트

### 최종 로그 (마지막 5줄)

```
cross_entropy: 0.51, L1: 0.063, Lpips: 0.033
cross_entropy: 0.55, L1: 0.087, Lpips: 0.041
cross_entropy: 0.12, L1: 0.036, Lpips: 0.018
cross_entropy: 0.42, L1: 0.061, Lpips: 0.029
Iteration finished.
```

---

## 6. 파일 수정 목록

| 파일 | 수정 내용 |
|------|-----------|
| `inference.py` | getMetaDict 필터 제거 (11,172자 전체 포함), weight 경로 변경 |
| `datasets/dataset_transformer.py` | 자모 유사도 대체 로직, 추론 시 augmentation 비활성화 |
| `trainer/trainer_utils.py` | discriminator 부분 복사 로직 (학습 시 shape mismatch 해결) |

---

*생성일: 2026-05-31*
