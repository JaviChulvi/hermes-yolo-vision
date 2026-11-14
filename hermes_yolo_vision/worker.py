"""Inference subprocess. Heavy imports and all model state live here."""

import hashlib
import io
import json
import os
import sys
import threading
import time
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .jobs import read_json, write_json
from .schemas import TRACK_TASKS


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observations(result):
    if result.depth is not None:
        import numpy as np

        depth = result.depth.data.cpu().numpy()
        valid = depth[np.isfinite(depth) & (depth > 0)]
        return {
            "depth": {
                "shape": list(depth.shape),
                "units": "estimated_meters",
                "valid_pixel_ratio": float(valid.size / depth.size),
                "min": float(valid.min()) if valid.size else None,
                "median": float(np.median(valid)) if valid.size else None,
                "max": float(valid.max()) if valid.size else None,
            }
        }
    return {"observations": result.summary(normalize=False)}


def save_raw(archive, result, index):
    import numpy as np

    arrays = {}
    for name, attribute in (
        ("instance_masks", "masks"),
        ("semantic_mask", "semantic_mask"),
        ("depth", "depth"),
    ):
        value = getattr(result, attribute, None)
        if value is not None:
            arrays[name] = value.data.cpu().numpy()
    if arrays:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        archive.writestr(f"frame-{index:08d}.npz", buffer.getvalue())


def run(record, directory):
    data = Path(record["data"])
    models = data / "models"
    models.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(data / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.chdir(models)

    import av
    import numpy as np
    from PIL import Image, ImageOps
    from PIL.Image import open as open_image
    from ultralytics import YOLO, YOLOE
    from ultralytics import __version__ as ultralytics_version

    args = record["arguments"]
    output = Path(record["output"])
    progress = directory / "progress.json"
    write_json(progress, {"phase": "preparing_model", "model": args["model"]})
    model = YOLO(args["model"])
    task = model.task
    requested = args.get("task")
    if args.get("text_prompts"):
        if not isinstance(model, YOLOE):
            raise ValueError("text_prompts require a YOLOE model")
        model.set_classes(args["text_prompts"])
        task = requested
    elif requested and requested != task:
        raise ValueError(f"Model task is {task}, but {requested} was requested")
    if record["operation"] == "yolo_track" and model.task not in TRACK_TASKS:
        raise ValueError(f"Tracking is unsupported for {model.task}")
    names = {int(k): v for k, v in model.names.items()}
    if record["operation"] == "yolo_info":
        query = args.get("class_query", "").casefold()
        matches = [{"id": k, "name": v} for k, v in names.items() if query in v.casefold()]
        return {
            "status": "completed",
            "task": task,
            "model": args["model"],
            "classes": matches[:100],
            "matching_classes": len(matches),
            "total_classes": len(names),
            "truncated": len(matches) > 100,
        }
    class_ids = None
    if args.get("classes"):
        by_name = {name.casefold(): key for key, name in names.items()}
        missing = [name for name in args["classes"] if name.casefold() not in by_name]
        if missing:
            raise ValueError(
                f"Classes unavailable in this model: {missing}. Use yolo_info or YOLOE text_prompts."
            )
        if task not in TRACK_TASKS:
            raise ValueError("Class filtering is supported for object tasks only")
        class_ids = [by_name[name.casefold()] for name in args["classes"]]
    size = args.get("imgsz", {"depth": 768, "classify": 224}.get(task, 640))
    predict = {
        "device": args["device"],
        "conf": args["conf"],
        "imgsz": size,
        "verbose": False,
        "stream": True,
        "save": False,
    }
    if class_ids is not None:
        predict["classes"] = class_ids
    if model.task == "segment":
        predict["retina_masks"] = True
    tracking = record["operation"] == "yolo_track"
    source = Path(args["source"])
    metadata = {
        "task": task,
        "model": args["model"],
        "ultralytics_version": ultralytics_version,
        "source_name": source.name,
        "source_sha256": sha256(source),
        "coordinate_space": "original_pixels_after_orientation",
        "parameters": args,
        "imgsz": size,
        "tracking": tracking,
        "audio": "not included in annotated video",
    }
    loaded_weights = Path(getattr(model, "ckpt_path", args["model"]))
    if loaded_weights.is_file():
        metadata["weights_sha256"] = sha256(loaded_weights)
    counts = Counter()
    track_ids = set()
    processed = 0
    decoded = 0
    preview = None
    artifacts = []
    raw = (
        zipfile.ZipFile(output / "raw.zip", "w", zipfile.ZIP_DEFLATED) if args["save_raw"] else None
    )
    observations_path = output / "observations.jsonl"

    def infer(image, index, timestamp, stream):
        nonlocal processed, preview
        results = (
            model.track(image, persist=True, tracker="bytetrack.yaml", **predict)
            if tracking
            else model.predict(image, **predict)
        )
        try:
            result = next(results)
        finally:
            results.close()
        frame = {
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "shape": list(result.orig_shape),
            **observations(result),
        }
        stream.write(json.dumps(frame, allow_nan=False) + "\n")
        rows = frame.get("observations", [])
        counts.update(row["name"] for row in rows if "box" in row)
        track_ids.update(row["track_id"] for row in rows if "track_id" in row)
        if preview is None:
            preview = {
                **frame,
                "observations": [
                    {k: v for k, v in row.items() if k != "segments"} for row in rows[:10]
                ],
            }
            preview["truncated"] = len(rows) > 10 or any("segments" in row for row in rows)
        if raw:
            save_raw(raw, result, index)
        processed += 1
        write_json(
            progress,
            {
                "phase": "inference",
                "processed_frames": processed,
                "last_frame": index,
                "timestamp_seconds": timestamp,
            },
        )
        return result.plot() if args["save_annotated"] else image

    try:
        with observations_path.open("w") as stream:
            if not args["video"]:
                with open_image(source, formats=["JPEG", "PNG", "WEBP", "BMP", "TIFF"]) as im:
                    image = np.array(ImageOps.exif_transpose(im).convert("RGB"))[:, :, ::-1].copy()
                plotted = infer(image, 0, None, stream)
                decoded = 1
                if args["save_annotated"]:
                    Image.fromarray(plotted[:, :, ::-1]).save(output / "annotated.jpg")
                    artifacts.append({"kind": "annotated", "path": str(output / "annotated.jpg")})
            else:
                writer = None
                try:
                    with av.open(str(source)) as container:
                        video = container.streams.video[0]
                        rate = video.average_rate
                        pixel_aspect = video.sample_aspect_ratio or Fraction(1)
                        metadata["source_sample_aspect_ratio"] = str(pixel_aspect)
                        if not rate or not video.time_base:
                            raise ValueError("Video has no usable frame rate/time base")
                        first_time = None
                        first_pts = None
                        for index, frame in enumerate(container.decode(video)):
                            if frame.pts is None or frame.time is None:
                                raise ValueError(
                                    "Video frame is missing its presentation timestamp"
                                )
                            if first_time is None:
                                first_time, first_pts = frame.time, frame.pts
                                metadata["source_start_seconds"] = first_time
                            timestamp = float(frame.time - first_time)
                            image = frame.to_ndarray(format="bgr24")
                            display = frame.side_data.get("DISPLAYMATRIX")
                            if display is not None:
                                matrix = np.frombuffer(display, dtype=np.int32).reshape(3, 3)
                                if int(matrix[0, 0]) * int(matrix[1, 1]) < int(matrix[0, 1]) * int(
                                    matrix[1, 0]
                                ):
                                    raise ValueError(
                                        "Mirrored video display metadata is unsupported; export an upright video before analysis"
                                    )
                            rotation = frame.rotation
                            if rotation % 90:
                                raise ValueError(
                                    "Video display rotation must be a multiple of 90 degrees"
                                )
                            if rotation:
                                image = np.rot90(image, k=round(rotation / 90)).copy()
                            height, width = image.shape[:2]
                            metadata["source_rotation_degrees"] = rotation
                            decoded += 1
                            if index % args["vid_stride"] == 0:
                                image = infer(image, index, timestamp, stream)
                            if args["save_annotated"]:
                                if writer is None:
                                    writer = av.open(str(output / "annotated.mp4"), mode="w")
                                    encoded = writer.add_stream("libx264", rate=rate)
                                    encoded.width = width + width % 2
                                    encoded.height = height + height % 2
                                    encoded.pix_fmt = "yuv420p"
                                    encoded.time_base = video.time_base
                                    encoded.codec_context.time_base = video.time_base
                                    encoded.codec_context.sample_aspect_ratio = (
                                        1 / pixel_aspect if rotation % 180 else pixel_aspect
                                    )
                                image = np.pad(image, ((0, height % 2), (0, width % 2), (0, 0)))
                                rendered = av.VideoFrame.from_ndarray(image, format="bgr24")
                                rendered.pts = frame.pts - first_pts
                                rendered.time_base = frame.time_base
                                for packet in encoded.encode(rendered):
                                    writer.mux(packet)
                        if writer:
                            for packet in encoded.encode():
                                writer.mux(packet)
                finally:
                    if writer:
                        writer.close()
                if args["save_annotated"] and decoded:
                    artifacts.append({"kind": "annotated", "path": str(output / "annotated.mp4")})
    finally:
        if raw:
            raw.close()
    if not processed:
        raise ValueError("No decodable frames in the supplied media")
    metadata.update(
        processed_frames=processed, decoded_frames=decoded, vid_stride=args["vid_stride"]
    )
    write_json(output / "metadata.json", metadata)
    artifacts.extend(
        [
            {"kind": "observations", "path": str(observations_path)},
            {"kind": "metadata", "path": str(output / "metadata.json")},
        ]
    )
    if raw:
        artifacts.append({"kind": "raw_arrays", "path": str(output / "raw.zip")})
    # JSONL is kept locally; a ZIP is recognized by every Hermes media adapter.
    with zipfile.ZipFile(output / "data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(observations_path, observations_path.name)
        archive.write(output / "metadata.json", "metadata.json")
    artifacts.append({"kind": "data", "path": str(output / "data.zip")})
    return {
        "status": "completed",
        "task": task,
        "model": args["model"],
        "processed_frames": processed,
        "decoded_frames": decoded,
        "vid_stride": args["vid_stride"],
        "detection_observations_by_class": dict(counts),
        "distinct_track_ids": len(track_ids) if tracking else None,
        "preview": preview,
        "artifacts": artifacts,
        "media": [
            f"MEDIA:{item['path']}"
            for item in artifacts
            if item["kind"] in ("annotated", "data", "raw_arrays")
        ],
        "interpretation": "Video counts are observations across analyzed frames, not unique objects. Track IDs and depth are estimates. Sampling may miss events.",
    }


def main():
    request = Path(sys.argv[1]).resolve()
    record = read_json(request)

    def parent_watch():
        while os.getppid() == record["parent_pid"]:
            time.sleep(1)
        os._exit(130)

    threading.Thread(target=parent_watch, daemon=True).start()
    try:
        payload = run(record, request.parent)
    except Exception as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(request.parent / "payload.json", payload)


if __name__ == "__main__":
    main()
