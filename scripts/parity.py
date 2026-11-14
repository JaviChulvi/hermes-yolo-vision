"""Compare smoke artifacts to direct Ultralytics results (serialization, not accuracy)."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            compare(a, e)
    elif isinstance(expected, (float, int)):
        assert np.isclose(actual, expected, atol=1e-4, rtol=1e-5), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(".agent/smoke-report.json"))
    parser.add_argument("--home", type=Path, default=Path(".agent/smoke-home"))
    options = parser.parse_args()
    report = json.loads(options.report.read_text())
    home = options.home.resolve()
    os.chdir(home / "plugin-data/hermes-yolo-vision/models")
    for case in report["cases"]:
        if case["text"] or case["video"]:
            continue
        result = case["result"]
        artifacts = {a["kind"]: Path(a["path"]) for a in result["artifacts"]}
        meta = json.loads(artifacts["metadata"].read_text())
        args = meta["parameters"]
        with Image.open(args["source"]) as source:
            image = np.array(ImageOps.exif_transpose(source).convert("RGB"))[:, :, ::-1].copy()
        model = YOLO(args["model"])
        direct = model.predict(image, imgsz=meta["imgsz"], conf=args["conf"], device=args["device"], retina_masks=True, verbose=False)[0]
        row = json.loads(artifacts["observations"].read_text())
        if model.task == "depth":
            depth = direct.depth.data.cpu().numpy()
            valid = depth[np.isfinite(depth) & (depth > 0)]
            assert np.isclose(row["depth"]["median"], np.median(valid))
        else:
            compare(row["observations"], direct.summary(normalize=False))
        print(json.dumps({"task": model.task, "parity": True}), flush=True)


if __name__ == "__main__":
    main()
