"""Fetch only the FSD50K clips of the 5 target classes into data/long_audio/<split>/<class>/<id>.wav.

Split mapping: FSD50K dev/train -> train, dev/val -> val, eval -> test.
Clips whose Freesound ID is also an ESC-50 source are not downloaded (pseudo_label.py would skip them anyway).
Audio comes from the Hugging Face mirror Fhrozen/FSD50k (one file per clip, unofficial); labels from the official
Zenodo zip. FSD50K has no baby-cry class, so crying_baby = Crying_and_sobbing (adult crying included).
Needs data/ESC-50-master (download_assets.py --esc50). Re-running skips clips already downloaded.

Usage: python fetch_fsd50k.py [project_root]    (about 1,250 clips, ~720 MB)
"""
import csv, io, sys, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
from common import esc50_source_ids  # noqa: E402

TARGETS = {"car_horn": "Vehicle_horn_and_car_horn_and_honking", "siren": "Siren",
           "door_wood_knock": "Knock", "crying_baby": "Crying_and_sobbing", "glass_breaking": "Shatter"}
GT_URL = "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1"
HF = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/{part}/{fname}.wav"

zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(GT_URL, timeout=120).read()))
esc = esc50_source_ids(ROOT / "data" / "ESC-50-master")
jobs, skipped = [], 0
for part, member in (("dev", "FSD50K.ground_truth/dev.csv"), ("eval", "FSD50K.ground_truth/eval.csv")):
    for r in csv.DictReader(io.TextIOWrapper(zf.open(member), encoding="utf-8")):
        labels = set(r["labels"].split(","))
        hits = [k for k, v in TARGETS.items() if v in labels]
        if not hits:
            continue
        if r["fname"] in esc:
            skipped += 1
            continue
        split = "test" if part == "eval" else ("train" if r["split"] == "train" else "val")
        dest = ROOT / "data" / "long_audio" / split / hits[0] / f"{r['fname']}.wav"
        jobs.append((HF.format(part=part, fname=r["fname"]), dest))
print(f"selected {len(jobs)} clips, skipped {skipped} ESC-50 overlaps", flush=True)


def get(job):
    url, dest = job
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
                f.write(resp.read())
            tmp.replace(dest)
            return 1
        except Exception as e:  # retry, then report
            err = e
    print("FAILED", url, err, flush=True)
    return -1


done = failed = 0
with ThreadPoolExecutor(16) as ex:
    for i, res in enumerate(ex.map(get, jobs), 1):
        done += res == 1
        failed += res == -1
        if i % 100 == 0:
            print(f"{i}/{len(jobs)} downloaded={done} failed={failed}", flush=True)
print(f"DONE downloaded={done} failed={failed} total={len(jobs)}", flush=True)
