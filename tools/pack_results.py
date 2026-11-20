"""실험 결과 폴더를 공유용 zip으로 묶기 (체크포인트 .pt 제외).

예: python tools/pack_results.py --experiment runs/noise_expansion_augmented_01
결과: runs/noise_expansion_augmented_01_results.zip

학습 완료 모델(.pt)은 수 GB라 기본으로 빼고, 결과 CSV/JSON/로그/그림/발표 자료
(presentation/ 의 WAV·PNG·index.html)를 모두 담습니다. 추론에 쓸 대표 모델 하나가
필요하면 --include-model noise_43/mixed 처럼 지정하세요(best.pt와 thresholds.json 포함).
"""
import argparse
import zipfile
from pathlib import Path


def pack(experiment, out=None, include_model=None):
    experiment = Path(experiment).resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(experiment)
    out = Path(out) if out else experiment.with_name(experiment.name + "_results.zip")
    keep_model = experiment / include_model / "best.pt" if include_model else None
    if keep_model is not None and not keep_model.is_file():
        raise FileNotFoundError(keep_model)
    files, skipped = [], 0
    for path in sorted(experiment.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".pt" and path != keep_model:
            skipped += 1
            continue
        files.append(path)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, Path(experiment.name) / path.relative_to(experiment))
    return out, len(files), skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--include-model", default=None, help="예: noise_43/mixed")
    a = p.parse_args()
    out, n, skipped = pack(a.experiment, a.out, a.include_model)
    print(f"Saved {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB; skipped {skipped} .pt checkpoints)")


if __name__ == "__main__":
    main()
