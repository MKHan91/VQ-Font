# VQ-Font 붓글씨 파인튜닝 분석 보고서

## 1. 현황 요약

| 항목 | 값 |
|------|-----|
| reference_images_v2 이미지 수 | **78장** |
| cr_mapping_v2.json 전체 유니코드 키 | **11,172개** |
| 한글 전체 글자 수 (목표) | **11,172자** |
| 부족한 구성요소 | 분석 필요 (78장으로 모든 초·중·종성 조합 커버 불가) |

> cr_mapping_v2.json은 각 한글 글자를 3개의 구성요소(hex 유니코드)로 매핑합니다.  
> 추론 시 해당 3개 구성요소 이미지가 모두 reference에 있어야만 해당 글자 생성 가능.

---

## 2. 추론 결과 부진 원인 분석

### 2-1. 데이터 문제 (가장 가능성 높음)

| 원인 | 설명 |
|------|------|
| **레퍼런스 이미지 수 부족** | reference_images_v2에 78자만 존재. cr_mapping의 모든 구성요소를 커버하지 못하여 추론 가능 글자가 제한됨. **11,172자 전체 생성 불가.** |
| **이미지 품질** | 스캔/촬영 노이즈, 해상도 불일치, 배경 처리 미비 시 학습 품질 저하. |
| **content reference 매핑 부족** | cr_mapping_v2.json에 모든 11,172자에 대한 분해 매핑이 정의되어야 함. 빠진 글자는 추론 자체가 불가능. |

### 2-2. 학습 설정 문제

| 원인 | 설명 |
|------|------|
| **50,000 iter 부족 가능성** | 붓글씨는 일반 폰트보다 획 변형이 커서 더 많은 학습 반복이 필요할 수 있음. 100,000~200,000 iter 권장. |
| **인코더 동결** | component_encoder와 content_encoder를 얼리면 붓글씨의 독특한 부품 형태를 새로 학습하지 못함. decoder만으로는 스타일 변환에 한계. |
| **75% 붓글씨 비율** | 나머지 25% 일반 폰트 학습이 기존 지식 유지 역할이지만, 붓글씨 학습을 방해할 수도 있음. 비율 조정 검토 필요. |

### 2-3. 추론 파이프라인 문제

| 원인 | 설명 |
|------|------|
| **체크포인트 선택** | inference.py에서 `generator_ema` → `generator` 순으로 로드. EMA가 더 안정적이므로 `generator_ema`가 없는 체크포인트는 품질 하락. |
| **kshot=3 참조 이미지** | 추론 시 참조 이미지 3장 사용. reference_images_v2에 해당 글자의 구성요소 이미지가 3장 모두 있어야 함. |
| **reduction='mean'** | 여러 참조를 평균내는 방식. 붓글씨처럼 변동이 큰 스타일에서는 특성이 뭉개질 수 있음. |

---

## 3. reference_images_v2 현재 글자 목록 (78자)

```
갉 값 걜 겨 곪 곬 곯 궜 궝 귤 기 깔 깡 껭 꽃 꿨 나 냅 냈 넵
뉠 닒 당 더 덮 돗 됐 듐 뜰 럇 렇 로 룟 마 맷 멂 몇 몲 무 밥
벰 볜 빎 빵 뽁 삥 삯 섧 솔 쏀 쓸 엶 옰 옳 옹 울 웸 위 젯 짝
짧 체 쳤 쵱 캔 켰 텅 텼 퉜 펙 포 퐝 푯 핥 행 헵 홍 흰
```

---

## 4. 권장 대응 순서

### 1단계: 레퍼런스 이미지 보강 ⭐ (최우선)
- 붓글씨 이미지를 **최소 200~300자**로 늘려 cr_mapping의 구성요소 커버율을 100%로 만들기
- `_check_coverage.py`를 실행하면 부족한 구성요소 목록 확인 가능
- 부족한 구성요소에 해당하는 글자 이미지를 추가해야 함

### 2단계: 학습 반복 수 증가
- `iter`를 **100,000 이상**으로 설정
- 붓글씨 스타일의 복잡한 획 변형을 충분히 학습하도록 함

### 3단계: 인코더 동결 해제 시도
- `component_encoder`에 매우 작은 LR(**1e-6**)로 미세 조정 허용
- 붓글씨 부품 인식 능력 개선

### 4단계: 중간 체크포인트 비교
- `save_freq=5000`마다 저장된 체크포인트 중 가장 좋은 결과 선택
- 과적합(overfitting) 방지

---

## 5. 파인튜닝 코드 검토 결과 (정상 확인)

| 파일 | 검토 결과 |
|------|-----------|
| `trainer_utils.py` | ✅ discriminator 부분 복사 로직 적용 완료. shape mismatch 시 기존 78개 임베딩 보존, 79번째(reference_images_v2) 랜덤 초기화. |
| `trainer_utils.py` | ✅ generator `strict=False` 로드 정상. optimizer/scheduler 미로드 (파인튜닝에 적합). |
| `train.py` | ✅ 인코더 동결 (component_encoder, content_encoder) 정상. optimizer에 `requires_grad=True` 파라미터만 포함. |
| `combined_trainer.py` | ✅ 학습 루프, D/G 분리 학습, gradient clipping 정상. |
| `base_trainer.py` | ✅ EMA (decay=0.999), save (generator_ema 포함), discriminator state_dict 저장 정상. |
| `custom_finetune.yaml` | ✅ LR, batch_size, iter, save_freq 등 설정 정상. save_freq(5000)가 val_freq(1000)의 배수 확인. |

---

## 6. 주요 설정값 (custom_finetune.yaml)

```yaml
batch_size: 16
iter: 50000
g_lr: 2e-5
d_lr: 8e-5
step_size: 2000
g_gamma: 0.95
d_gamma: 0.98
kshot: 3
cv_n_fonts: 1      # 붓글씨 폰트 1개
cv_n_unis: 10
val_freq: 1000
save_freq: 5000
```

---

*생성일: 2026-05-29*
