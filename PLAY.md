# PLAY — 긴 클린 음원으로 라벨링 → 학습 → 평가 → 추론

이 문서는 **사람이 할 일**만 순서대로 적었습니다. 명령은 프로젝트 폴더(`audio_noise_project`)에서 PowerShell로 실행합니다.

## 한눈에 보기

| 단계 | 누가 | 무엇을 | 결과가 생기는 곳 |
|---|---|---|---|
| 0 | 사람 | 환경·ESC-50·사전학습 가중치 준비 (최초 1번) | `.venv/`, `data/ESC-50-master/`, `checkpoints/` |
| 1 | 사람 | 깨끗한 긴 음원을 폴더에 넣기 | `data/long_audio/` |
| 2 | 코드 | 선생님 기준값 만들기 (최초 1번, 이미 있으면 생략) | `runs/teacher_calibration/` |
| 3 | 코드 | 원본 AudioSet Cnn14가 5초 창마다 라벨 붙이기 | `data/long_audio_labels/` |
| 4 | 사람 | 라벨 확인·수정 (특히 test) | `data/long_audio_labels/windows.csv` |
| 5 | 코드 | 학습 목록(manifest) 만들기 | `data/manifest_long.json` |
| 6 | 코드 | 학습 | `runs/long_01/` |
| 7 | 코드 | 검증(기준값 결정) → 테스트 | `runs/long_01/validation/`, `runs/long_01/test/` |
| 8 | 코드 | 새 음원 추론 | `runs/long_infer_01/` |

> GPU가 없으면 6~8단계 명령 끝에 `--device cpu`를 붙이세요. (설정이 `cuda`라서 자동으로 CPU로 바뀌지 않습니다.) 2~3단계는 GPU가 있으면 자동으로 씁니다.

---

## 0. 최초 1번: 환경 준비

README.md의 "B. CUDA 실행 환경 만들기", "C. 원본 데이터와 사전학습 가중치 받기"를 따릅니다. 끝나면 아래 두 개가 있어야 합니다.

- `data/ESC-50-master/` (소음 섞기와 기본 학습 데이터에 계속 사용)
- `checkpoints/Cnn14_mAP=0.431.pth` (라벨을 붙이는 "선생님" 모델이자 학습 출발점)

## 1. 음원 준비 (사람)

### 어떤 음원?
- 목표음 5종(경적, 사이렌, 문 두드림, 아기 울음, 유리 깨짐)이 **들어 있는 깨끗한 녹음**.
- **소음을 일부러 섞지 마세요.** 소음은 학습할 때 코드가 자동으로 섞습니다. 선생님 모델은 깨끗한 소리에서 정확하고, 시끄러운 소리에서는 많이 놓치기 때문입니다.
- 길이 제한은 없습니다(몇 초 ~ 몇십 분). 목표음이 녹음 내내 있을 필요도 없습니다. 창마다 따로 라벨을 붙입니다.
- 형식: wav, flac, ogg, mp3. 샘플레이트·스테레오 상관없음(자동 변환).
- 목표음이 **전혀 없는** 생활 녹음(방, 거리 등)도 있으면 `none` 폴더에 넣으세요. "없음" 예시로 쓰입니다.

### 어디에?
```
data/long_audio/
├─ train/                ← 학습용 (약 70%)
│  ├─ car_horn/          ← 폴더 이름 = 이 녹음의 주된 목표음
│  ├─ siren/
│  ├─ door_wood_knock/
│  ├─ crying_baby/
│  ├─ glass_breaking/
│  └─ none/              ← 목표음 없는 녹음
├─ val/                  ← 검증용 (약 15%, 같은 구조)
└─ test/                 ← 최종 평가용 (약 15%, 같은 구조)
```
- 폴더 이름은 위 6개 중 하나여야 합니다. 다른 곳에 파일이 있으면 3단계에서 위치를 알려주며 멈춥니다.
- **같은 녹음에서 잘라낸 조각은 반드시 한 split에만** 넣으세요. (같은 장소·같은 날 녹음이 train과 test에 나뉘면 점수가 부풀려집니다.) 파일 이름이 split 사이에 겹쳐도 멈춥니다.
- val/test가 비어 있어도 됩니다. 그러면 기존 ESC-50 검증/평가만으로 점수를 냅니다.
- 폴더 이름(클래스)은 **점검용**입니다. 실제 라벨은 선생님 모델이 창마다 정합니다. 예를 들어 사이렌 녹음에 경적이 섞여 있으면 그 창은 경적도 1이 될 수 있습니다.

## 2. 선생님 기준값 (최초 1번, 코드)

`runs/teacher_calibration/calibration.json`이 이미 있으면 **건너뜁니다.** 없을 때만:

```powershell
.\.venv\Scripts\python.exe pseudo_label.py calibrate
```
```powershell
.\.venv\Scripts\python.exe pseudo_label.py check
```

- `calibrate`: ESC-50 fold 1~4의 깨끗한 클립으로 클래스별 기준 두 개를 정합니다. "이 값 이상이면 1(95% 이상 맞음)", "이 값 미만이면 0(진짜 목표음을 5% 이하로만 놓침)".
- `check`: 기준 정할 때 안 쓴 fold 5로 라벨 정확도를 확인합니다. 결과는 `runs/teacher_calibration/check_test_fold.json`.

## 3. 라벨링 (코드)

```powershell
.\.venv\Scripts\python.exe pseudo_label.py label
```

- `data/long_audio/`의 모든 파일을 5초 창(2.5초 간격)으로 나눠 원본 AudioSet Cnn14로 판단합니다.
- 창마다, 클래스마다 **1(있음) / 0(없음) / -1(애매함)** 이 붙습니다. -1은 학습·평가에서 자동으로 빠집니다.
- 결과 폴더 `data/long_audio_labels/`
  - `windows.csv`: 창별 표 (파일, 시작·끝 초, 클래스별 점수 `p_*`, 라벨 `label_*`, 선생님이 들은 상위 3개 소리 `teacher_top3`)
  - `files.csv`: 파일별 요약과 `needs_review`(확인 필요) 표시
  - `meta.json`: 사용한 기준값·가중치 기록
- 결과 폴더가 이미 있으면 **덮어쓰지 않고 멈춥니다**(손으로 고친 라벨 보호). 새로 하려면 `--out data/long_audio_labels_02`처럼 새 이름을 주세요.

## 4. 라벨 확인 (사람) — 중요

1. `files.csv`에서 `needs_review = 1`인 파일부터 봅니다. 이유가 `review_reason`에 있습니다.
   - "no window labeled 1 for siren": 사이렌 폴더인데 선생님이 사이렌을 한 번도 확신하지 못함 → 직접 들어보고 해당 창을 1로 고치거나, 정말 없으면 파일을 빼세요.
   - "teacher found a target in a 'none' recording": 목표음 없다던 녹음에서 목표음을 찾음 → 들어보고 맞으면 두고, 틀리면 0으로.
2. **test 파일은 모든 창을 직접 들어보고 확정하세요.** 선생님 라벨로 테스트하면 "선생님과 얼마나 같은가"만 재게 됩니다. val도 가능하면 확인합니다.
3. train은 needs_review 파일과 일부 무작위 창만 확인해도 충분합니다.
4. 고치는 법: `windows.csv`를 엑셀로 열어 `label_클래스` 칸을 1/0/-1로 바꾸고 **CSV(UTF-8)** 로 저장. 시간은 `start_seconds`~`end_seconds` 구간입니다. -1을 그대로 두면 그 칸은 학습에서 무시됩니다.

## 5. 학습 목록 만들기 (코드)

```powershell
.\.venv\Scripts\python.exe prepare.py --esc50 "data/ESC-50-master" --long-labels "data/long_audio_labels" --out "data/manifest_long.json"
```

- 기존 ESC-50 학습 데이터에 긴 음원의 창들을 **더합니다.**
- 긴 음원 창 하나마다: 깨끗한 버전 1개 + ESC-50 소음을 섞은 버전(train 1개, val/test는 학습에 없던 소음 포함 2개).
- 출력 마지막 줄의 `long-recording windows:` 숫자로 추가된 양을 확인합니다.

## 6. 학습 (코드)

```powershell
.\.venv\Scripts\python.exe train.py --mode mixed --manifest "data/manifest_long.json" --out "runs/long_01"
```

- 매번 새 이름의 `--out`을 쓰세요(있으면 멈춥니다).
- 매 epoch 소음·SNR·크기를 새로 뽑는 증강 학습을 원하면 5·6단계 모두에 `--config config_augmented.json`을 붙이세요.
- 결과 모델: `runs/long_01/best.pt`

## 7. 검증 → 테스트 (코드)

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint "runs/long_01/best.pt" --manifest "data/manifest_long.json" --split val --tune --out "runs/long_01/validation"
```
```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint "runs/long_01/best.pt" --manifest "data/manifest_long.json" --split test --thresholds "runs/long_01/validation/thresholds.json" --out "runs/long_01/test"
```

- val에서 클래스별 판정 기준을 정하고, test는 그 기준을 그대로 써서 점수만 냅니다.
- `report.json`의 `conditions`에서 `long/...`로 시작하는 항목이 긴 음원 창만의 점수입니다.

## 8. 추론 (코드)

```powershell
.\.venv\Scripts\python.exe infer.py --audio "새음원.wav" --checkpoint "runs/long_01/best.pt" --thresholds "runs/long_01/validation/thresholds.json" --out "runs/long_infer_01"
```

- 파일 길이와 상관없이 5초 창을 1초 간격으로 훑어 창마다 판정합니다(`windows.csv`).
- "이 파일에 사이렌이 있나?" = 어느 한 창이라도 사이렌으로 검출되었는가.
- 시간은 창의 범위이지 정확한 시작·끝 시각이 아닙니다.

## 새 음원이 더 생기면

1단계 폴더에 추가 → 3단계를 **새 출력 폴더**로 실행(예: `--out data/long_audio_labels_02`, 이전에 고친 라벨은 옛 폴더에 남음) → 4단계 확인 → 5~7단계를 새 이름으로 반복.

## 자주 막히는 곳

| 메시지 | 뜻 / 할 일 |
|---|---|
| `Files must be at <split>/<class>/<file>` | 음원 위치·폴더 이름이 틀림. 1단계 구조 확인 |
| `Calibration missing` | 2단계 `calibrate`를 먼저 실행 |
| `Output folder not empty` / `Already exists` | 출력 폴더가 이미 있음. 새 이름 사용 |
| `Same file name in several splits` | 같은 녹음이 여러 split에 들어감. 한 곳으로 옮기기 |
| `CUDA requested but unavailable` | GPU 없음. 명령 끝에 `--device cpu` |
| `Manifest/config mismatch` | 5단계와 6단계의 `--config`가 다름. 같은 설정으로 다시 5단계 |

## 알아둘 한계

- 선생님(원본 AudioSet Cnn14)이 틀리면 그 실수도 학습됩니다. 그래서 4단계 사람 확인이 품질을 좌우합니다.
- 선생님 기준값은 ESC-50 클립(클래스당 32개)으로 정했습니다. 실제 녹음 환경이 많이 다르면 정확도가 달라질 수 있으니, 처음 몇 개 파일은 결과를 들어보며 확인하세요.
- 선생님 라벨의 실제 정확도(기준 정할 때 안 쓴 ESC-50 fold 5, 깨끗한 클립, 클래스당 목표음 8개):

| 클래스 | 진짜 있음 → 1 | → -1 | → 0 (놓침) | 진짜 없음인데 → 1 (오탐) |
|---|---:|---:|---:|---:|
| 자동차 경적 | 100% | 0% | 0% | 0.5% |
| 사이렌 | 88% | 0% | 12% | 0% |
| 문 두드림 | 75% | 0% | 25% | 0.3% |
| 아기 울음 | 88% | 12% | 0% | 0% |
| 유리 깨짐 | 75% | 12% | 12% | 0% |

  - **1은 거의 믿을 수 있습니다**(오탐 0.5% 이하).
  - **약점은 놓침**입니다. 진짜 목표음인데 0이 붙는 경우가 있습니다(목표음 8개 중 1~2개). 그래서 목표음 폴더 파일의 0 창, 특히 소리가 시작·끝나는 경계 창은 들어볼 가치가 있습니다.
  - 목표음 8개로 잰 값이라 1개 차이가 12%p입니다. 대략적인 경향으로만 보세요.
