"""Audit files independently; emit blinded evidence, never an automatic success verdict."""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import av
import numpy as np
from PIL import Image


def neutralize(value):
    if isinstance(value, str):
        return re.sub(r"""(?<![\w:/])(?:MEDIA:)?/[^\s<>"']+""", "[artifact reference]", value)
    if isinstance(value, list):
        return [neutralize(item) for item in value]
    if isinstance(value, dict):
        return {neutralize(key): neutralize(item) for key, item in value.items()}
    return value


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def objects(value):
    if isinstance(value, list):
        for item in value:
            yield from objects(item)
    elif isinstance(value, dict):
        name = next(
            (value[k] for k in ("name", "label", "class_name") if isinstance(value.get(k), str)),
            None,
        )
        if name and any(
            k in value
            for k in ("box", "bbox", "bbox_xyxy", "keypoints", "confidence", "score", "probability")
        ):
            yield value
        else:
            for item in value.values():
                yield from objects(item)


def array_stats(stream):
    with np.load(stream, allow_pickle=False) as data:
        result = {}
        for key in data.files:
            values = data[key]
            if values.dtype.kind not in "biuf":
                result[key] = {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "preview": values.ravel()[:20].astype(str).tolist(),
                }
                continue
            finite = values[np.isfinite(values)]
            result[key] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "finite_fraction": float(finite.size / values.size) if values.size else 0,
                "positive_fraction": float((np.isfinite(values) & (values > 0)).mean())
                if values.size
                else 0,
                "min": float(finite.min()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
            }
        return result


def inspect(path):
    result = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp"):
        with Image.open(path) as image:
            result.update(kind="image", size=list(image.size), mode=image.mode)
            image.verify()
    elif suffix in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        with av.open(str(path)) as video:
            count = 0
            first = last = None
            monotonic = True
            for frame in video.decode(video=0):
                if first is None:
                    first = frame.time
                    result["size"] = [frame.width, frame.height]
                if last is not None and frame.time <= last:
                    monotonic = False
                last = frame.time
                count += 1
            result.update(
                kind="video",
                frames=count,
                first_seconds=first,
                last_seconds=last,
                strictly_increasing_timestamps=monotonic,
            )
    elif suffix == ".npz":
        result.update(kind="arrays", arrays=array_stats(path))
    elif suffix == ".csv":
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        result.update(
            kind="table", rows=len(rows), columns=list(rows[0]) if rows else [], preview=rows[:3]
        )
    elif suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            assert bad is None, f"Corrupt ZIP member {bad}"
            result.update(kind="zip", members=archive.namelist())
            arrays = [n for n in archive.namelist() if n.endswith(".npz")]
            if arrays:
                result["first_arrays"] = array_stats(io.BytesIO(archive.read(arrays[0])))
    elif suffix in (".json", ".jsonl"):
        rows = (
            [
                json.loads(line, parse_constant=reject_constant)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            if suffix == ".jsonl"
            else json.loads(path.read_text(), parse_constant=reject_constant)
        )
        rows = rows if isinstance(rows, list) else [rows]
        found = list(objects(rows))
        result.update(kind="structured", rows=len(rows), object_rows=len(found), preview=found[:12])
        timestamps = [
            r["timestamp_seconds"]
            for r in rows
            if isinstance(r, dict) and isinstance(r.get("timestamp_seconds"), (int, float))
        ]
        if timestamps:
            result.update(
                first_seconds=timestamps[0],
                last_seconds=timestamps[-1],
                strictly_increasing_timestamps=all(
                    b > a for a, b in zip(timestamps, timestamps[1:])
                ),
            )
    else:
        result["kind"] = "unrecognized-inspect-manually"
    return result


def audit_run(root, options):
    study, identifier = root.parent, root.name
    run = json.loads((root / "result.json").read_text())
    request = json.loads((root / "request.json").read_text())
    turns = run.get("turns", [])
    audit = {
        "review_id": identifier,
        "prompt": request["case"]["prompt"],
        "followup": request["case"].get("followup"),
        "rubric": request["case"].get("rubric"),
        "execution_status": run["status"],
        "completed_turns": [t.get("completed", False) for t in turns],
        "final_answers": [],
        "artifacts": [],
        "grading": "pending",
    }
    for turn in turns:
        answer = turn.get("final_response", "")
        audit["final_answers"].append(answer)
    manifest = root / "work/handoff.json"
    if not manifest.exists():
        audit["manifest_error"] = (
            "No handoff manifest; inspect final-answer references before grading."
        )
    else:
        try:
            handoff = json.loads(manifest.read_text())
            for index, item in enumerate(handoff["artifacts"]):
                path = Path(item["path"])
                if options.source_root and path.is_absolute():
                    path = study / path.relative_to(options.source_root)
                path = path if path.is_absolute() else root / "work" / path
                path = path.resolve()
                assert path.is_relative_to(root), "Artifact belongs outside this run"
                evidence = {"id": f"artifact-{index}", "declared_kind": item["kind"]}
                try:
                    evidence.update(inspect(path))
                    if options.copy_artifacts:
                        neutral = options.output / identifier / (evidence["id"] + path.suffix)
                        neutral.parent.mkdir(exist_ok=True)
                        shutil.copyfile(path, neutral)
                        evidence["review_file"] = str(neutral.relative_to(options.output))
                except Exception as exc:
                    evidence["error"] = f"{type(exc).__name__}: {exc}"
                audit["artifacts"].append(evidence)
        except Exception as exc:
            audit["manifest_error"] = f"{type(exc).__name__}: {exc}"
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Copy artifacts under neutral names for visual review",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Original study root when inspecting a copied evidence bundle",
    )
    options = parser.parse_args()
    study = options.study.resolve()
    options.output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((study / "runs.json").read_text())
    for key, identifier in mapping.items():
        root = study / identifier
        if not (root / "result.json").exists():
            continue
        try:
            audit = audit_run(root, options)
            encoded = json.dumps(neutralize(audit), indent=2, allow_nan=False)
        except Exception as exc:
            encoded = json.dumps(
                {
                    "review_id": identifier,
                    "audit_error": neutralize(f"{type(exc).__name__}: {exc}"),
                    "grading": "pending",
                }
            )
        (options.output / (identifier + ".json")).write_text(encoded + "\n")
    print(
        json.dumps(
            {
                "evidence_files": len(list(options.output.glob("*.json"))),
                "grading": "pending: apply RUBRIC.md, including ground-truth overlap and answer checks",
            }
        )
    )


if __name__ == "__main__":
    main()
