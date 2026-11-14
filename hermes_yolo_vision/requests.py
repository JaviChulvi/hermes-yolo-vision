"""Validate inputs before starting an expensive or downloading operation."""

import math
import re
from pathlib import Path

from .schemas import DEFAULT_MODELS, SCHEMAS, TEXT_MODEL, TRACK_TASKS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
OFFICIAL = re.compile(
    r"(?:yolo(?:26|11|v8)[nslmx](?:-seg|-pose|-cls|-obb|-sem(?:-ade20k)?|-depth)?|yoloe-26[nslmx]-seg)\.pt\Z"
)


def validate(tool, args, cwd, read_check=lambda path: None, device="cpu"):
    if not isinstance(args, dict):
        raise ValueError("Arguments must be an object")
    spec = next(s["parameters"] for s in SCHEMAS if s["name"] == tool)
    unknown = args.keys() - spec["properties"].keys()
    if unknown:
        raise ValueError(f"Unknown arguments: {', '.join(sorted(unknown))}")
    for name in spec["required"]:
        if name not in args:
            raise ValueError(f"Missing argument: {name}")
    for name, value in args.items():
        prop = spec["properties"][name]
        kind = prop["type"]
        valid = {
            "string": isinstance(value, str),
            "boolean": type(value) is bool,
            "number": type(value) in (int, float),
            "integer": type(value) is int,
            "array": isinstance(value, list),
        }[kind]
        if not valid:
            raise ValueError(f"{name} must be {kind}")
        if kind == "string" and (not value.strip() or len(value) > 4096):
            raise ValueError(f"{name} must be a nonempty, bounded string")
        if kind in ("number", "integer"):
            if not math.isfinite(value) or not prop.get("minimum", -math.inf) <= value <= prop.get(
                "maximum", math.inf
            ):
                raise ValueError(f"{name} is outside its allowed range")
        if "enum" in prop and value not in prop["enum"]:
            raise ValueError(f"Unsupported {name}: {value}")
        if kind == "array" and (
            not prop["minItems"] <= len(value) <= prop["maxItems"]
            or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in value)
        ):
            raise ValueError(f"{name} must contain a bounded list of nonempty class names")
    out = dict(args)
    if tool == "yolo_job":
        if not re.fullmatch(r"[a-f0-9]{32}", out["job_id"]):
            raise ValueError("Invalid job_id")
        return out

    def local(value):
        path = Path(value).expanduser()
        path = (Path(cwd) / path).resolve() if not path.is_absolute() else path.resolve()
        error = read_check(str(path))
        if error:
            raise ValueError(error)
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path

    if "source" in out:
        if "://" in out["source"]:
            raise ValueError("Download the attachment first; v1 accepts local files only")
        path = local(out["source"])
        if path.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            raise ValueError("Unsupported image/video file format")
        out["source"] = str(path)
        out["video"] = path.suffix.lower() in VIDEO_EXTS
        if tool == "yolo_track" and not out["video"]:
            raise ValueError("Tracking requires a video file")
    if "model" in out and not OFFICIAL.fullmatch(out["model"]):
        path = local(out["model"])
        if path.suffix != ".pt":
            raise ValueError("Custom models must be trusted local .pt checkpoints")
        out["model"] = str(path)
    if out.get("text_prompts"):
        if out.get("classes") or out.get("task", "segment") not in ("detect", "segment"):
            raise ValueError(
                "Text prompts support detect/segment and cannot be combined with classes"
            )
        out.setdefault("task", "segment")
        out.setdefault("model", TEXT_MODEL)
    if tool != "yolo_info":
        if "model" not in out:
            out.setdefault("task", "detect")
            out["model"] = DEFAULT_MODELS[out["task"]]
        out.setdefault("conf", 0.25)
        out.setdefault("vid_stride", 1)
        out.setdefault("save_raw", False)
        out.setdefault("save_annotated", True)
    out.setdefault("device", device)
    if not re.fullmatch(r"cpu|mps|\d+", str(out["device"])):
        raise ValueError("device must be cpu, mps, or a CUDA index")
    if tool == "yolo_track" and out.get("task", "detect") not in TRACK_TASKS:
        raise ValueError("This task cannot be tracked")
    return out
