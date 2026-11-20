"""Download original-author source; optionally download checkpoint and ESC-50."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = "qiuqiangkong/panns_inference"
DEFAULT_REVISION = "f673f604ec6f4805a61c5b3be087e24776ec5fda"


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        print("Already exists:", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "audio-noise-course-project"})
    print("Downloading:", url, flush=True)
    with urllib.request.urlopen(req, timeout=120) as response, temporary.open("wb") as f:
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
    temporary.replace(destination)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", action="store_true", help="Download standard 32kHz Cnn14 pretrained checkpoint (~310 MiB)")
    p.add_argument("--esc50", action="store_true", help="Download ESC-50 (~600MB) and extract into data/ESC-50-master")
    p.add_argument("--checkpoint-path", default=str(ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"))
    p.add_argument("--revision", default=None, help="Optional pinned upstream Git commit; existing bundled source is reused by default")
    a = p.parse_args()
    vendor = ROOT / "third_party" / "panns"
    source_info = vendor / "SOURCE.json"
    if source_info.exists() and a.revision is None:
        info = json.loads(source_info.read_text(encoding="utf-8"))
        for filename, expected in info["sha256"].items():
            if digest(vendor / filename) != expected:
                raise RuntimeError(f"Bundled source changed: {filename}")
        print("Verified bundled source revision:", info["revision"])
    else:
        revision = a.revision or DEFAULT_REVISION
        if source_info.exists():
            raise RuntimeError("Source already installed; use a fresh project directory to change revisions")
        hashes = {}
        for filename, remote in (("models.py", "panns_inference/models.py"),
                                 ("pytorch_utils.py", "panns_inference/pytorch_utils.py"),
                                 ("LICENSE", "LICENSE.MIT")):
            download(f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{remote}", vendor / filename)
            hashes[filename] = digest(vendor / filename)
        # Do not use upstream __init__: it downloads labels to the user's home at import time.
        (vendor / "__init__.py").write_text('"""Original PANNs implementation. See SOURCE.json and LICENSE."""\n', encoding="utf-8")
        (vendor.parent / "__init__.py").write_text("", encoding="utf-8")
        source_info.write_text(json.dumps({"repository": f"https://github.com/{REPOSITORY}",
                               "revision": revision, "sha256": hashes}, indent=2), encoding="utf-8")
    if a.weights:
        download("https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1", a.checkpoint_path)
        print("Checkpoint SHA256:", digest(a.checkpoint_path))
    if a.esc50:
        archive = ROOT / "data" / "ESC-50-master.zip"
        download("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", archive)
        destination = (ROOT / "data").resolve()
        if (destination / "ESC-50-master").exists():
            print("ESC-50 directory already exists; extraction skipped")
            return
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise RuntimeError("Unsafe ZIP entry")
            z.extractall(destination)
        print("ESC-50:", destination / "ESC-50-master")


if __name__ == "__main__":
    main()
