"""Download evaluation media and freeze provenance/checksums before agent runs."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import av
from cases import CASES, DEV_CASES
from ultralytics.utils import ASSETS

BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    opts = parser.parse_args()
    root = opts.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    origins = {}
    for name in ("coco8.zip", "dota8.zip", "Pushups.demo.video.mp4", "Squats.demo.video.mp4", "Benchpress.demo.video.mp4"):
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(BASE + name, path)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    target = (root / member.filename).resolve()
                    if not target.is_relative_to(root):
                        raise ValueError("Archive member escapes fixture directory")
                archive.extractall(root)
        origins[name] = {"url": BASE + name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    files = {}
    for n in (25, 30, 34, 36, 42, 49, 61):
        files[f"coco{n}"] = next((root / "coco8/images").glob(f"*/{n:012d}.jpg"))
    files["dota-a"] = root / "dota8/images/train/P1161__1024__3296___1648.jpg"
    files["dota-b"] = root / "dota8/images/val/P1571__1024__2976___0.jpg"
    files["pushups"] = root / "Pushups.demo.video.mp4"
    files["squats"] = root / "Squats.demo.video.mp4"
    files["benchpress"] = root / "Benchpress.demo.video.mp4"
    files["squat-frame"] = root / "squat-frame.jpg"
    with av.open(str(files["squats"])) as video:
        next(video.decode(video=0)).to_image().save(files["squat-frame"])
    files["bus"] = root / "bus.jpg"
    files["bus"].write_bytes((ASSETS / "bus.jpg").read_bytes())
    files["corrupt"] = root / "corrupt.jpg"
    files["corrupt"].write_bytes(b"Deliberately invalid image fixture for recovery evaluation.\n")
    files["dev-corrupt"] = root / "dev-corrupt.jpg"
    files["dev-corrupt"].write_bytes(b"Development fixture: not a JPEG.\n")
    manifest = {
        "sources": origins,
        "provenance_notes": "COCO8 and DOTA8 retain their dataset/image licenses; see upstream dataset documentation. Fitness media are upstream public demo assets; downloaded for testing, not redistributed. Bus is the installed Ultralytics sample. Corrupt file is authored here. Squat-frame is frame zero of the unmodified squat clip. Demo clips with burned-in annotations are excluded.",
        "fixture_labels": {"pushups": {"complete_repetitions": 4, "completion_seconds_approx": [2, 4, 6, 8], "method": "Assistant visual inspection of the raw clip timeline before model runs; not human-adjudicated"}, "squats": {"complete_repetitions": 4, "completion_seconds_approx": [2.2, 4, 5.7, 7.3], "method": "Assistant visual inspection of the raw clip timeline before model runs; not human-adjudicated"}},
        "files": {key: {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in files.items()},
        "cases": CASES,
        "development_cases": DEV_CASES,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fixtures": len(files), "heldout_cases": len(CASES), "development_cases": len(DEV_CASES), "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
