# 생활소음 인식 시스템

음성 및 오디오 처리 과목의 2인 팀 프로젝트입니다. 청각장애인용 디스플레이형 보조안경을 목표로, 5초 음원 구간에 아래 목표음 5종이 있는지 PANNs Cnn14 전이학습으로 판별합니다. 현재는 PC에서 파일 입력으로 동작합니다.

## 먼저 읽기

- **AI 에이전트로 작업을 이어받는다면** [CLAUDE.md](CLAUDE.md)를 먼저 읽게 하세요. 규칙과 현재 남은 과제가 정리돼 있습니다.
- **긴 클린 음원으로 라벨링 → 학습 → 추론**을 하려면 [PLAY.md](PLAY.md)를 보세요.
- **발표·보고서용 수치와 해석 주의사항**은 [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)에 있습니다.

꼭 구분할 것:

1. **본 실험**은 `runs/noise_expansion_20261115/`입니다. 방해음 6·18·30·43종 × clean/mixed, 총 8모델이고 검증 650구간, 평가 652구간을 공통으로 씁니다.
2. `runs/week1_20261115/`은 2모델 **예비 실험**입니다. 평가가 356구간이라 본 실험 점수와 직접 비교하면 안 됩니다.
3. 루트 `config.json`은 **기본 6종 방해음 설정**입니다. 본 실험 단계별 설정은 `runs/noise_expansion_20261115/noise_XX/config.json`에 있습니다.
4. RMS 무음 검사는 본 실험에 적용하지 않았습니다. 실시간 마이크 검출은 미구현입니다.
5. ESC-50 원본(`data/ESC-50-master/`), 가중치(`checkpoints/`), 학습 결과(`runs/`), `.venv`는 `.gitignore` 대상이라 따로 받아야 합니다. `data/`의 나머지(AudioSet 라벨 목록, 긴 음원, 자동 라벨)는 git에 포함됩니다.
6. 저장된 manifest의 음원 경로는 작성자 PC의 절대경로입니다. 다른 PC에서는 **새 출력 폴더에 manifest를 다시 만드세요**.

## 1. 목적과 고정 범위

출력 순서는 코드, 레이블, 점수, 체크포인트, 임계값에서 모두 같습니다.

| 출력 인덱스 | ESC-50 클래스 | 표시 이름 |
|---|---|---|
| 0 | car_horn | 자동차 경적 |
| 1 | siren | 사이렌 |
| 2 | door_wood_knock | 문 두드림 |
| 3 | crying_baby | 아기 울음 |
| 4 | glass_breaking | 유리 깨짐 |

정규 학습·평가에는 **목표음 최대 한 종류**만 씁니다. 목표음 단독, 목표음 + 비대상 방해음, 방해음 단독, 인위적 무음입니다. 목표음이 없으면 레이블은 `[0,0,0,0,0]`입니다. 출력이 독립 sigmoid라 여러 개가 동시에 기준을 넘을 수는 있지만, 중첩 인식을 검증했다는 뜻은 아닙니다.

다음은 범위 밖입니다: 목표음 파형 분리, 방향 추정, 정확한 사건 시작·종료 시각, 안경 하드웨어 연동.

## 2. 파일 구성

```text
audio_noise_project/
  README.md  CLAUDE.md  PLAY.md      안내서 / AI 에이전트용 규칙 / 긴 음원 작업 절차
  config.json                        기본 설정 (6종 방해음)
  config_augmented.json              학습 증강을 켠 설정 (11절)
  requirements.txt                   설치용 의존성 범위
  requirements-cuda-lock.txt         실제 CUDA 실행 환경의 버전 기록
  download_assets.py                 PANNs 소스 확인, ESC-50·사전학습 가중치 다운로드

  # 데이터 준비
  prepare.py                         ESC-50 원본 분할·혼합 recipe → manifest (--long-labels로 긴 음원 창 추가)
  noise_expansion.py                 본 실험 4단계 데이터 구성과 학습·검증·평가 일괄 실행
  pseudo_label.py                    원본 AudioSet Cnn14로 긴 음원 5초 창 자동 라벨링 (PLAY.md)
  audio_utils.py                     파형 읽기·리샘플링·패딩·혼합·RMS, recipe → 파형
  augment.py                         학습 전용 증강 (매 epoch 재혼합, 위치 이동, SpecAugment 등)

  # 모델·학습·평가·추론
  model.py                           Cnn14 특징 추출 + 새 5출력 분류기
  train.py                           2단계 전이학습, -1(불확실) 라벨 무시하는 masked BCE
  metrics.py                         지표 계산과 소리별 기준 선택
  evaluate.py                        검증 기준 조정, 고정 기준 평가, 선택적 RMS 검사
  infer.py                           긴 오디오 파일의 이동 구간 추론

  # 결과 정리·발표 자료
  plot_results.py                    예비 실험 2모델 비교 그래프
  plot_noise_expansion.py            본 실험 8모델 취합과 29개 그래프
  analyze_sigmoid.py                 저장 점수의 분포·SNR·PR·기준 변화 분석
  presentation_assets.py             발표용 예시 음원·판정 카드·그래프 (12절)
  analysis/baseline_pretrained_cnn14.py  학습 없는 사전학습 Cnn14와 우리 모델 비교
  analysis/layer_intuition.py        Cnn14 층별 특징 분석 그림
  tools/pack_results.py              실험 결과 폴더를 공유용 zip으로 묶기 (.pt 제외)
  run_experiments.ps1                기본 2모델 실험 실행 도우미

  tests/                             unittest (파형·지표·모델·통합·증강·긴 음원)
  third_party/panns/                 원저자 Cnn14 소스, MIT LICENSE, SOURCE.json
  docs/EXPERIMENT_LOG.md             실험 기록과 발표용 수치·해석 주의
  docs/RMS_DESIGN.md                 연속 입력용 RMS 게이트 설계안 (미구현)
  docs/dataset_audit.json            ESC-50 무결성·분할 검사 기록
  docs/ESC50_LICENSE.txt             ESC-50 이용 조건 (CC BY-NC)

  data/audioset_class_labels_indices.csv  AudioSet 527 라벨 목록 (pseudo_label.py가 사용)
  data/long_audio/, data/long_audio_labels/   긴 음원과 자동 라벨 (PLAY.md)

  # .gitignore 대상 (따로 받거나 생성)
  data/ESC-50-master/                ESC-50 원본 약 1.4 GB (download_assets.py --esc50)
  data/manifest*.json                학습 목록. 그 PC의 절대경로가 들어가므로 prepare.py로 다시 생성
  checkpoints/Cnn14_mAP=0.431.pth    AudioSet 사전학습 가중치
  runs/noise_expansion_20261115/     본 실험 결과
  runs/week1_20261115/               예비 실험 결과
  runs/baseline_pretrained_01/       사전학습 Cnn14 비교 결과
  runs/layer_intuition_01/           층별 분석 그림
  runs/teacher_calibration/          긴 음원 라벨링 기준값 (PLAY.md 2단계)
```

권장 코드 읽기 순서: `config.json` → `audio_utils.py` → `model.py` → `prepare.py` → `noise_expansion.py` → `train.py` → `metrics.py` → `evaluate.py` → `infer.py` → `augment.py` → `pseudo_label.py`

### 데이터 흐름

- **준비**: ESC-50 메타데이터 → 원본 단위 분할(fold 1~3 학습, 4 검증, 5 평가) → 혼합 recipe를 담은 manifest. 긴 음원은 `pseudo_label.py` → 사람 확인 → `prepare.py --long-labels`로 창 단위 행을 추가합니다. 파일 이름의 Freesound 번호가 ESC-50 녹음(전체 2,000개)과 같은 긴 음원은 두 단계 모두에서 제외합니다. FSD50K도 Freesound에서 왔기 때문에 같은 녹음이 학습과 평가 양쪽에 들어가는 것을 막기 위해서입니다.
- **학습**: `RecipeDataset` → `render_recipe()` → `[batch, 160000]` 파형 → Cnn14 → `[batch, 5]` logits → masked BCE(-1 라벨 제외).
- **평가**: 최종 모델 → sigmoid 점수 → 검증에서 고른 소리별 기준 → 지표. -1 라벨 칸은 지표에서도 제외합니다.
- **파일 추론**: 5초 구간을 기본 1초 간격으로 이동 → (선택) RMS 검사 → Cnn14 → 구간별 CSV/JSON. 시간값은 분석 구간이지 사건의 정확한 경계가 아닙니다.

## 3. 전처리·모델 구현 상세

- 파형은 float32 모노로 읽습니다(다채널은 평균). `resample_poly`로 32 kHz로 바꾸고 5초(160,000샘플)로 맞춥니다. 짧으면 뒤를 0으로 채웁니다.
- 목표음 RMS를 0.03~0.15로 조정하고 SNR 0·5·10 dB로 방해음을 섞습니다. SNR은 실제 울리는 부분이 아니라 **5초 전체 RMS** 기준입니다.
- 혼합 후 최대 절댓값이 0.95를 넘으면 전체를 같은 비율로 줄입니다(하드 클리핑 아님, SNR 유지).
- 무음 12구간은 같은 0 파형의 반복입니다.
- Cnn14 내부 Log-Mel: FFT/window 1024, hop 320, Mel 64대역, 50~14,000 Hz(약 32 ms 분석, 10 ms 이동).
- 2048차원 임베딩에 `nn.Linear(2048,5)`를 새로 붙였습니다. 학습은 logits로, 평가·추론에서만 sigmoid를 씁니다. 저장된 `scores`는 이미 sigmoid이므로 다시 적용하지 마세요. 확률 보정은 하지 않았습니다.
- 앞쪽 블록은 고정합니다. 분류기 10 epoch 후 최우수 상태를 복원하고, `conv_block6`·`fc1`·head를 5 epoch 미세조정합니다. 전체 15 epoch 중 검증 손실이 최소인 가중치를 고릅니다.
- backbone은 eval 모드로 두어 BatchNorm 통계를 고정하고 Dropout을 끕니다. eval 모드여도 미세조정 대상 파라미터는 학습됩니다.
- CUDA AMP에서 Log-Mel이 float16 underflow를 일으켜, Spectrogram·Log-Mel·입력 BatchNorm은 float32로 유지하고 CNN과 head에만 AMP를 씁니다.

## 4. 본 실험 결과 요약

| 학습 방해음 종류 | 방해음 학습 원본 | 학습 recipe 수 | clean 평가 F1 | mixed 평가 F1 | mixed 목표음 없는 구간 오검출 |
|---|---:|---:|---:|---:|---:|
| 6 | 144 | 636 | 0.812007 | 0.868839 | 32/372 |
| 18 | 432 | 924 | 0.832647 | 0.918594 | 14/372 |
| 30 | 720 | 1212 | 0.845244 | 0.919255 | 12/372 |
| 43 | 1032 | 1524 | 0.838095 | 0.914096 | 11/372 |

- 학습 입력: 깨끗한 목표음 120 + 혼합용 목표음 360 + 방해음 단독 + 무음 12. clean 방식은 혼합용 360개도 목표음만 제시해서 두 방식의 노출 횟수를 맞춥니다.
- 검증 650구간(목표음 40 + 혼합 240 + 방해음 358 + 무음 12), 평가 652구간(40 + 240 + 360 + 12). 소리별 양성은 각 56개입니다.
- washing_machine과 footsteps는 전이학습에서 제외하고 검증·평가에만 넣은 방해음입니다.
- 결과를 볼 파일: `summary.json`, `data_size_metrics.csv`, `all_epoch_metrics.csv`, `figures/`, `sigmoid_analysis/`.

해석 주의:

- 검출 기준은 모델마다 검증에서 따로 고릅니다(예: 30종 mixed는 `[0.80,0.20,0.15,0.60,0.35]`). 다른 모델에 재사용하면 안 됩니다.
- Macro-F1은 5개 소리 F1의 평균입니다. 50종 분류 정확도가 아닙니다.
- "목표음 없는 구간 오검출"은 그 구간에서 어느 목표음이든 잘못 검출한 경우입니다.
- 같은 목표음 원본을 여러 혼합 조건에 썼으므로 652구간이 독립 녹음 652개는 아닙니다.
- 시드 42 한 번의 결과입니다. 방해음 종류·단독 비율·업데이트 수가 함께 변하므로 데이터 양 하나의 인과 효과가 아닙니다.
- 평가 결과를 보고 모델이나 기준을 다시 고르지 않습니다.

예비 실험, 사전학습 Cnn14 비교, 층별 분석, 선생님 모델 기준값 결과는 [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)에 있습니다.

## 5. 조원 PC에서 시작하는 방법

### A. 결과만 볼 때

설치 없이 소스, Markdown, CSV/JSON, PNG를 열면 됩니다. `runs/` 결과는 작성자에게 zip(`tools/pack_results.py`)으로 받습니다.

### B. CUDA 실행 환경 만들기

기록된 환경: Windows, Python 3.11.15, torch 2.7.1+cu128, RTX 4060 Ti 8GB, batch 4. 조원 PC의 NVIDIA 드라이버와 Python 호환성을 확인하세요. 설정의 장치는 `cuda`이고 CPU로 자동 전환하지 않습니다(GPU가 없으면 명령에 `--device cpu`).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

당시 패키지 버전을 정확히 맞추려면(Windows/Python 3.11/CUDA 12.8 기준):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

### C. 원본 데이터와 사전학습 가중치 받기

```powershell
.\.venv\Scripts\python.exe download_assets.py --weights --esc50
```

`data/ESC-50-master/`와 `checkpoints/Cnn14_mAP=0.431.pth`가 생깁니다. **사전학습 가중치는 5종 검출 모델이 아닙니다.** 바로 추론하려면 작성자에게 `best.pt`와 그에 맞는 `thresholds.json`을 받거나 재학습해야 합니다.

## 6. 실행 방법

### 본 실험(8모델) 재실행

새 출력 폴더 이름을 쓰세요. 기존 결과를 덮어쓰지 않습니다. 자동 재개 기능은 없습니다.

```powershell
# 1) 4단계 구성과 manifest 생성
.\.venv\Scripts\python.exe noise_expansion.py --esc50 "data/ESC-50-master" --out "runs/team_expansion_01" --prepare-only
# 2) 4단계 × clean/mixed 학습·검증·평가 (끝나면 12절의 후처리도 자동 실행)
.\.venv\Scripts\python.exe noise_expansion.py --out "runs/team_expansion_01" --run-prepared
```

후처리를 따로 다시 돌릴 때:

```powershell
.\.venv\Scripts\python.exe plot_noise_expansion.py --experiment "runs/team_expansion_01"
.\.venv\Scripts\python.exe analyze_sigmoid.py --experiment "runs/team_expansion_01" --out "runs/team_expansion_01/sigmoid_analysis"
```

`plot_noise_expansion.py`는 `best.pt`를 읽어 검증하므로, 체크포인트가 없는 기존 결과 폴더에는 실행되지 않습니다. `analyze_sigmoid.py`는 저장된 예측만 쓰므로 음원·`.pt` 없이도 됩니다.

### 기본 설정 2모델 (prepare → train → evaluate)

```powershell
.\.venv\Scripts\python.exe prepare.py --esc50 data/ESC-50-master --out data/manifest.json
.\.venv\Scripts\python.exe train.py --mode mixed --manifest data/manifest.json --out runs/mixed_01
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/mixed_01/best.pt --manifest data/manifest.json --split val --tune --out runs/mixed_01/validation
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/mixed_01/best.pt --manifest data/manifest.json --split test --thresholds runs/mixed_01/validation/thresholds.json --out runs/mixed_01/test
```

`--mode clean`으로 바꾸면 깨끗한 학습입니다. `run_experiments.ps1 -Esc50Root data/ESC-50-master`는 이 과정을 두 방식 모두 실행하고 비교 그래프까지 만듭니다. `train.py --record-metrics`는 epoch별 검증 지표를 기록합니다.

RMS 검사를 켜서 평가하려면 `evaluate.py`에 `--rms-db -60`을 붙입니다(예시 값이며 최적값 아님, 검증에서 먼저 정할 것).

### 파일 추론

```powershell
.\.venv\Scripts\python.exe infer.py --audio "demo.wav" --checkpoint "runs/team_expansion_01/noise_30/mixed/best.pt" --thresholds "runs/team_expansion_01/noise_30/mixed/validation/thresholds.json" --out "runs/demo_file_01"
```

검출 기준 파일은 모델 SHA-256과 클래스 순서에 연결돼 있습니다. 다른 모델의 기준 파일을 섞거나 해시를 고쳐 통과시키지 마세요.

### 긴 클린 음원 작업

[PLAY.md](PLAY.md)를 따르세요. (`pseudo_label.py calibrate/label` → 사람 확인 → `prepare.py --long-labels` → `train.py` → `evaluate.py` → `infer.py`)

## 7. 저장 결과의 파일 역할

| 파일 | 읽는 목적 |
|---|---|
| plan.json | 단계 순서, 클래스, 중복 제외, 검증/평가 지문 |
| noise_XX/config.json | 해당 단계의 실제 설정 |
| noise_XX/manifest.json | 고정 혼합 recipe와 분할 (당시 PC 절대경로) |
| clean 또는 mixed/history.csv | epoch별 train/val 손실과 검증 지표 |
| training_summary.json | 최저 검증 손실과 실행 시간 |
| validation/thresholds.json | 검증에서 고른 소리별 기준과 모델 해시 |
| validation 또는 test/predictions.csv | 구간별 정답·sigmoid 점수·RMS 제외 여부 |
| report.json, per_class.csv | 전체/소리별/조건별 평가 |
| provenance.json | 라이브러리·GPU·입력·가중치 출처 해시 |
| all_epoch_metrics.csv | 8모델 총 120 epoch 취합 |
| data_size_metrics.csv, summary.json | 8모델 최종 비교 |
| verification.json, source_hashes.json | 당시 데이터·코드 검증 기록 |
| sigmoid_analysis/ | 저장 점수의 추가 분석 CSV/JSON과 그래프 7개 |

CSV의 리스트 점수는 Python literal 형식이므로 `ast.literal_eval`로 읽습니다.

## 8. 테스트와 미구현 사항

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

합성 파형·SNR·지표·분할·RMS·시간 구간, 실제 Cnn14 구조와 역전파, 작은 대체 모델로 파이프라인 통합, 증강, 긴 음원 라벨·masked BCE를 검사합니다. 테스트 통과가 실제 환경 성능을 보장하지는 않습니다.

미구현·미검증:

- 실시간 마이크 입력과 화면 알림 시연. `infer.py`는 파일 입력 전용입니다.
- 긴 실제 음원에서 시간당 오경보, 같은 사건의 반복 알림 처리.
- RMS 패딩 제외 개선과 RMS 게이트의 실제 시간 절감 효과(설계안: `docs/RMS_DESIGN.md`).
- 두 목표음 중첩, 방향 추정, 안경 연동.
- 다중 시드, 다른 환경의 추가 원본으로 본 일반화 성능.

## 9. 출처

- PANNs 소스 라이선스와 커밋/해시: `third_party/panns/LICENSE`, `SOURCE.json`.
- ESC-50: https://github.com/karolpiczak/ESC-50 (이용 조건 사본 `docs/ESC50_LICENSE.txt`, 음원은 포함하지 않음).
- 공식 모델: https://github.com/qiuqiangkong/audioset_tagging_cnn, https://github.com/qiuqiangkong/panns_inference
- 사전학습 가중치: https://zenodo.org/records/3987831
- AudioSet 라벨 목록(`data/audioset_class_labels_indices.csv`)은 `pseudo_label.py`가 사용합니다.

## 10. 다른 데이터로 넓힐 때

- 목표음 5종의 ESC-50 원본은 합계 200개뿐입니다. 혼합본을 늘려도 원본 다양성은 늘지 않습니다.
- 긴 환경 녹음은 같은 녹음을 여러 분할에 나누지 말고 **녹음 원본 단위로** 분할하세요. 방해음 녹음에 목표음이 이미 들어 있지 않은지 들어서 확인하세요.

## 11. 학습 증강 (`augment.py`, `config_augmented.json`)

본 실험에는 쓰지 않았습니다. 설정에 `augment` 항목이 없으면 이전과 똑같이 동작합니다. **학습(train) 분할에만** 적용되고 검증·평가는 고정 recipe를 그대로 쓰므로 기존 점수와 같은 기준으로 비교됩니다.

| 기능 | 설정 키 | `config_augmented.json` 값 |
|---|---|---|
| 매 epoch 새로 섞기(방해음·SNR·음량·위치를 (seed, epoch, 항목)으로 재추첨) | `enabled` | `true` |
| 시작 위치 변환(뒤쪽 0 패딩을 잘라내고 무작위 위치에 배치, 에너지 70% 이상 유지) | `time_shift`, `min_target_energy_keep` | `true`, `0.7` |
| 연속 SNR | `snr_range_db` | `[-5, 15]` |
| 목표음 음량, 방해음 단독 음량 | `target_rms_range`, `negative_gain_db` | 기존 범위, `±6 dB` |
| 시간 일부 겹치기(방해음이 창의 일부만 덮음) | `partial_overlap_prob`, `partial_overlap_min_fraction` | `0.3`, `0.3` |
| 긴 원본 겹침 crop(5초보다 긴 원본을 1초 간격 slot으로 나눠 무작위 시작) | `long_crop_hop_seconds`, `max_crops_per_source` | `1.0`, `30` |
| 긴 목표음 crop 검사(파일 최대 프레임 대비 20 dB 안의 소리가 있는 crop만 양성) | `long_target_peak_db` | `20` |
| SpecAugment(학습 중에만, BatchNorm은 계속 고정) | `spec_augment` | 시간 32프레임×2, 주파수 8bin×2 |

```powershell
.\.venv\Scripts\python.exe noise_expansion.py --config config_augmented.json --out runs/noise_expansion_augmented_01 --prepare-only
.\.venv\Scripts\python.exe noise_expansion.py --out runs/noise_expansion_augmented_01 --run-prepared
```

주의:

- `pseudo_label.py`로 라벨을 붙인 긴 음원 창(`source: "long"` 행)은 라벨이 그 창에 묶여 있으므로 crop·위치 이동을 하지 않습니다. 음량과 방해음만 다시 뽑습니다.
- 라벨 없이 train 행에 긴 파일을 직접 넣으면 crop이 적용되는데, 이때는 "목표음이 그 녹음에서 가장 큰 소리"라고 가정합니다.
- 원본 반복 횟수가 달라지므로 기존 실험과 비교할 때는 epoch 수와 업데이트 수를 함께 보고하세요.

## 12. 결과·발표 자료 자동 저장

`noise_expansion.py --run-prepared`는 학습·검증·평가 후 아래를 자동 실행합니다. 이 단계가 실패해도 학습 결과는 남고, 상태는 `execution_status.json`의 `post_processing`에 `ok` 또는 `failed: ...`로 기록됩니다.

| 단계 | 스크립트 | 저장 위치 |
|---|---|---|
| 8모델 그래프 취합 | `plot_noise_expansion.py` | `figures/`, `summary.json` 등 |
| sigmoid 점수 분석 | `analyze_sigmoid.py` | `sigmoid_analysis/` |
| 발표용 자료 | `presentation_assets.py` | `presentation/` |

### `presentation/` 내용

대표 모델은 **검증 Macro-F1이 가장 높은 모델**입니다(`--model noise_43/mixed`로 지정 가능).

- `index.html`: 아래 자료를 브라우저 한 화면에서 보고 듣기.
- `audio/by_snr/`: 같은 녹음을 깨끗한 소리, 10·5·0 dB로 저장한 예시(소리마다 "모두 맞힌 예시"와 "어려운 예시", 방해음만 있는 `*_noise_only.wav` 포함). 모델이 실제로 들은 파형입니다.
- `figures/decisions/`: 판정 카드(위 스펙트로그램, 아래 5개 점수와 기준선, 제목에 정답·놓침·오인·오검출).
- `figures/recall_by_snr.png`, `confusion.png`, `false_alarms_by_noise.png`, `training_curve.png`.
- `figures/compare_models.png`, `compare_recall_by_snr.png`: 기존 실험과 같은 평가셋으로 비교.
- `figures/augmentation_examples.png`, `spec_augment_example.png`, `audio/augmentation/`: 증강이 무엇을 바꾸는지.
- `audio/errors/`, `errors.csv`: 기준에서 가장 크게 벗어난 놓침·오검출 사례.
- `summary.json`, `examples.csv`: 수치 요약. 기존 실험 대비 Macro-F1 차이의 95% 신뢰구간 포함(원본 녹음 단위 bootstrap).

용량은 약 40 MB이고 대부분 WAV입니다.

### 결과 주고받기

```powershell
.\.venv\Scripts\python.exe tools/pack_results.py --experiment "runs/noise_expansion_augmented_01"
```

`.pt`를 뺀 `runs/noise_expansion_augmented_01_results.zip`이 생깁니다. 시연용 모델 하나를 같이 보내려면 `--include-model noise_43/mixed`를 붙입니다.

받은 PC에서 발표 자료를 다시 만들 때는 체크포인트나 GPU 없이 ESC-50 음원만 있으면 됩니다. 다른 PC의 manifest 경로는 `--audio-root`의 파일명으로 찾습니다.

```powershell
.\.venv\Scripts\python.exe presentation_assets.py --experiment "runs/noise_expansion_augmented_01" --out "runs/noise_expansion_augmented_01/presentation_v2"
```
