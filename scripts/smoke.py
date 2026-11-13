"""Real inference and artifact checks; synthetic clips test plumbing, not model accuracy."""

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import av
import numpy as np
from PIL import Image
from ultralytics.utils import ASSETS

from hermes_yolo_vision.jobs import Jobs, write_json
from hermes_yolo_vision.requests import validate
from hermes_yolo_vision.schemas import DEFAULT_MODELS


def static_clip(image, destination):
    from fractions import Fraction

    pixels = np.array(Image.open(image).convert("RGB").resize((640, 480)))
    with av.open(str(destination), "w") as output:
        stream = output.add_stream("libx264", rate=6)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 6)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def check_result(result, task, expected_frames, raw=False):
    assert result["status"] == "completed", result
    assert result["processed_frames"] == expected_frames, result
    artifacts = {a["kind"]: Path(a["path"]) for a in result["artifacts"]}
    for path in artifacts.values():
        assert path.is_file() and path.stat().st_size > 0, path
    rows = [json.loads(line) for line in artifacts["observations"].read_text().splitlines()]
    assert len(rows) == expected_frames
    first = rows[0]
    if task == "depth":
        assert first["depth"]["valid_pixel_ratio"] > 0.9 and first["depth"]["median"] > 0
    else:
        assert first["observations"], f"Empty payload does not certify {task}"
        field = {"segment": "segments", "pose": "keypoints", "semantic": "pixel_ratio", "classify": "confidence", "obb": "box", "detect": "box"}[task]
        assert field in first["observations"][0]
    if raw:
        with zipfile.ZipFile(artifacts["raw_arrays"]) as z:
            assert z.namelist()
            with z.open(z.namelist()[0]) as f:
                with np.load(f) as arrays:
                    assert arrays.files and all(arrays[k].size for k in arrays.files)
    annotation = artifacts["annotated"]
    if annotation.suffix == ".mp4":
        with av.open(str(annotation)) as video:
            frames = list(video.decode(video=0))
            assert len(frames) == 6 and frames[-1].time > frames[0].time
    else:
        with Image.open(annotation) as image:
            image.verify()
    return {"task": task, "frames": expected_frames, "artifacts_checked": len(artifacts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=Path(".agent/smoke-home"))
    parser.add_argument("--report", type=Path, default=Path(".agent/smoke-report.json"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--tracking", action="store_true")
    opts = parser.parse_args()
    home = opts.home.resolve()
    fixtures = home / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    if "obb" in opts.tasks:
        archive = fixtures / "dota8.zip"
        if not archive.exists():
            urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip", archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.endswith((".jpg", ".png")) and "/images/" in name:
                    z.extract(name, fixtures)
        obb = sorted(fixtures.glob("dota8/images/**/*.jpg")) + sorted(fixtures.glob("dota8/images/**/*.png"))
    jobs = Jobs(home / "plugin-data/hermes-yolo-vision", home / "cache/generated/hermes-yolo-vision")
    report = {"scope": "Real inference; video fixtures repeat a static image and do not demonstrate real-world tracking accuracy.", "cases": []}
    cases = [(task, "yolo_predict", {}) for task in opts.tasks]
    if opts.text:
        cases.append(("segment", "yolo_predict", {"text_prompts": ["person", "bus"]}))
    if opts.tracking:
        cases.append(("pose", "yolo_track", {}))
    try:
        for task, operation, extra in cases:
            source = obb[0] if task == "obb" else ASSETS / "bus.jpg"
            video = opts.video or operation == "yolo_track"
            if video:
                clip = fixtures / f"{task}.mp4"
                static_clip(source, clip)
                source = clip
            raw = task in ("segment", "semantic", "depth")
            args = validate(operation, {"source": str(source), "task": task, "save_raw": raw, **extra}, Path.cwd())
            start = time.monotonic()
            result = jobs.submit(operation, args, "smoke")
            if result["status"] == "running":
                result = jobs.get(result["job_id"], "smoke", 900)
            case = {"task": task, "operation": operation, "text": bool(extra), "video": video, "elapsed_seconds": time.monotonic() - start, "result": result}
            try:
                case["checks"] = check_result(result, task, 6 if video else 1, raw)
                if operation == "yolo_track":
                    assert result["distinct_track_ids"] > 0
                case["passed"] = True
            except Exception as exc:
                case.update(passed=False, failure=str(exc))
            report["cases"].append(case)
            write_json(opts.report, report)
            print(json.dumps({k: case[k] for k in ("task", "operation", "video", "passed", "elapsed_seconds")}), flush=True)
    finally:
        jobs.close()
    return 0 if all(c["passed"] for c in report["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
