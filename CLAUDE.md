# CLAUDE.md — AI 에이전트용 인수인계

이 파일은 이 프로젝트를 이어받는 AI 에이전트(Claude Code 등)가 가장 먼저 읽는 문서입니다. 사람에게 답할 때는 한국어로 쓰세요.

## 프로젝트 한 줄 요약

PANNs Cnn14(AudioSet 사전학습, 32 kHz)를 전이학습해 5초 음원 구간에서 목표음 5종(car_horn, siren, door_wood_knock, crying_baby, glass_breaking)의 존재 여부를 판별합니다. 수업 2인 팀 프로젝트이고 발표가 남아 있습니다.

## 읽는 순서

1. `README.md`: 전체 구조, 실행 명령, 본 실험 결과와 해석 주의
2. `PLAY.md`: 긴 클린 음원 → 선생님 모델 자동 라벨 → 학습 → 추론 절차 (가장 최근 작업)
3. `docs/EXPERIMENT_LOG.md`: 발표용 수치와 "말하면 안 되는 것"
4. `docs/RMS_DESIGN.md`: 연속 입력용 RMS 게이트 설계안 (미구현)

## 실행 환경

- Windows, PowerShell. 파이썬은 항상 `.\.venv\Scripts\python.exe`를 씁니다.
- 테스트: `.\.venv\Scripts\python.exe -m unittest discover -s tests` (약 35개, 1개 skip, 1분 안팎)
- `config.json`의 `device`는 `cuda`이고 CPU로 자동 전환하지 않습니다. GPU가 없으면 `--device cpu`.
- `.gitignore` 대상: `data/ESC-50-master/`(+zip), `data/manifest*.json`, `checkpoints/`, `runs/`, `.venv/`. ESC-50과 가중치는 `download_assets.py --weights --esc50`로 받고, 결과(`runs/`)는 사람에게 zip으로 받습니다.
- `data/`의 나머지는 git에 포함됩니다: `audioset_class_labels_indices.csv`(pseudo_label.py 필수), `long_audio/`(빈 분류 폴더는 `.gitkeep`으로 유지), `long_audio_labels/`. 긴 음원 파일이 커지면 Git LFS를 검토하세요(GitHub는 100 MB 넘는 파일을 거부).

## 깨면 안 되는 규칙

- **클래스 순서** `[car_horn, siren, door_wood_knock, crying_baby, glass_breaking]`는 코드·레이블·점수·체크포인트·기준값 모두에서 같아야 합니다.
- **검출 기준값은 검증(val)에서만** 고릅니다(`evaluate.py --split val --tune`). test로 기준이나 모델을 고르지 마세요. 코드도 이를 막습니다.
- **기존 `runs/` 결과를 덮어쓰지 마세요.** 특히 `runs/noise_expansion_20261115/`(발표 본 실험)와 `runs/teacher_calibration/`. 새 실험은 항상 새 폴더로 만듭니다.
- `thresholds.json`은 모델 SHA-256에 묶여 있습니다. 해시를 고쳐서 다른 모델에 쓰지 마세요.
- 저장된 `scores`는 이미 sigmoid입니다. 다시 sigmoid를 적용하지 마세요.
- 정규 학습·평가에서 한 창에 목표음은 최대 한 종류입니다. 예외는 긴 음원 행(`source: "long"`)뿐입니다.
- 라벨 `-1`은 "불확실"입니다. 손실(`train.masked_bce`)과 지표(`metrics.scores`, `tune_thresholds`)에서 제외됩니다. -1을 0이나 1로 바꾸는 코드를 넣지 마세요.
- 긴 음원 창은 라벨이 그 5초에 묶여 있으므로 증강에서 crop·위치 이동을 하지 않습니다.
- 학습 원본 분할은 ESC-50 fold 1~3 학습, 4 검증, 5 평가입니다. 같은 원본 녹음이 여러 분할에 들어가면 안 됩니다.
- 긴 음원(FSD50K·Freesound)은 파일 이름 앞의 Freesound 번호(`common.freesound_id`)가 ESC-50의 `src_file`과 같으면 제외합니다. `pseudo_label.py label`이 건너뛰고, `prepare.add_long_rows`가 한 번 더 막습니다. 이 검사를 끄거나 파일 이름을 바꿔서 우회하지 마세요.

## 사실과 계획을 구분할 것

구현·실측된 것과 계획을 섞지 마세요. 아래는 **미구현**입니다: 실시간 마이크 검출, RMS 게이트(설계만 있음), RMS 패딩 제외 개선, 두 목표음 중첩, 방향 추정, 안경 연동, 다중 시드 실험. 발표에서 쓸 수 있는 표현은 `docs/EXPERIMENT_LOG.md`를 따르세요.

## 현재 상태 (2026-11-20 기준)

- 본 실험 8모델, 예비 실험 2모델, 사전학습 Cnn14 비교, 층별 분석이 끝났습니다.
- 긴 클린 음원 경로(`pseudo_label.py` → `prepare.py --long-labels` → `train.py`/`evaluate.py` → `infer.py`)를 구현했고 합성 데이터로 끝까지 돌아가는 것을 확인했습니다. **실제 긴 음원으로는 아직 학습하지 않았습니다.** 긴 음원 후보는 FSD50K이며, ESC-50과 겹치는 Freesound 녹음은 자동으로 제외됩니다.
- 학습 증강(`augment.py`, `config_augmented.json`)은 구현했지만 본 실험에 쓰지 않았습니다.

## 남은 과제 (코드 리뷰에서 나온 미수정 항목, 중요한 순서)

1. **긴 음원 SNR 기준**: `audio_utils.render_long_window`와 `augment.render_augmented`의 긴 음원 분기가 소음 크기를 창 자신이 아니라 **파일에서 가장 큰 창** 기준으로 정합니다. 목표음이 작은 창은 이름상 0 dB여도 훨씬 낮은 SNR로 묻히는데 라벨은 1로 남습니다. 수정안: 라벨이 1인 창은 창 자신의 RMS를 SNR 기준으로 씁니다.
2. `pseudo_label.find_audio`는 `.wav/.flac/.ogg/.mp3` 외 형식(.m4a 등)을 **경고 없이 건너뜁니다**.
3. `prepare.add_long_rows`는 `windows.csv`를 utf-8-sig로만 읽습니다. 한국어 엑셀이 cp949로 저장하면 한글 경로에서 깨집니다.
4. `windows.csv`의 `path`가 절대경로라 다른 PC로 옮기면 수동 수정한 라벨을 쓸 수 없습니다. `--audio-root` 기준 상대경로로 바꾸는 것이 좋습니다.
5. "같은 파일 이름이 여러 split에 있음" 검사가 이름만 같은 다른 녹음을 막고, 이름을 바꾼 같은 녹음은 놓칩니다(내용 해시 검사 권장).
6. `pseudo_label label`은 결과를 마지막에 한 번에 씁니다. 중간에 깨진 파일 하나로 전체가 실패하고, 창이 하나도 없으면 `rows[0]`에서 IndexError가 납니다.
7. `label_report`가 NaN을 낼 수 있는데 `write_json(allow_nan=False)`에서 오류가 납니다.
8. `source == "long"` 분기가 여러 파일에 흩어져 있습니다(정리 대상).
9. 라벨링할 때 겹치는 창을 따로 읽고 리샘플링해서 같은 구간을 두 번씩 처리합니다(효율).

고친 뒤에는 테스트를 다시 돌리고, `tests/test_long_audio.py`에 해당 경우를 추가하세요.

## 작업 습관

- 큰 학습은 오래 걸립니다(8모델 약 40분 이상). 사람이 요청하지 않으면 학습을 자동으로 돌리지 마세요.
- 문서와 코드가 다르면 코드가 기준입니다. 다른 점을 찾으면 문서를 고치세요.
- 새 기능을 넣으면 README의 파일 구성(2절)과 이 파일의 "현재 상태"를 같이 갱신하세요.
