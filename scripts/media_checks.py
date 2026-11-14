"""Real inference checks for video timing/orientation, EXIF, custom weights, and errors."""

import argparse
import json
import shutil
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image
from ultralytics.utils import ASSETS

from hermes_yolo_vision.jobs import Jobs
from hermes_yolo_vision.requests import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(".agent/smoke-home"))
    opts = parser.parse_args()
    home = opts.home.resolve()
    fixtures = home / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    clip = fixtures / "rotated-vfr.mp4"
    pixels = np.array(Image.open(ASSETS / "bus.jpg").resize((640, 480)))
    with av.open(str(clip), "w") as out:
        stream = out.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.sample_aspect_ratio = Fraction(3, 2)
        stream.set_display_rotation(90)
        for pts in [5000, 5100, 5350, 5400, 5900, 6100]:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    exif_path = fixtures / "exif.jpg"
    image = Image.open(ASSETS / "bus.jpg")
    exif = image.getexif()
    exif[274] = 6
    image.save(exif_path, exif=exif)
    corrupt = fixtures / "broken.jpg"
    corrupt.write_bytes(b"not an image")
    model_dir = home / "plugin-data/hermes-yolo-vision/models"
    custom = fixtures / "custom.pt"
    shutil.copy2(model_dir / "yolo26n.pt", custom)
    jobs = Jobs(
        home / "plugin-data/hermes-yolo-vision", home / "cache/generated/hermes-yolo-vision"
    )

    def run(args):
        args = validate("yolo_predict", args, Path.cwd())
        result = jobs.submit("yolo_predict", args, "media-check")
        if result["status"] == "running":
            result = jobs.get(result["job_id"], "media-check", 120)
        return result

    try:
        result = run({"source": str(clip), "vid_stride": 2})
        assert result["status"] == "completed", result
        paths = {a["kind"]: Path(a["path"]) for a in result["artifacts"]}
        rows = [json.loads(line) for line in paths["observations"].read_text().splitlines()]
        assert [r["frame_index"] for r in rows] == [0, 2, 4]
        with av.open(str(clip)) as source:
            assert source.streams.video[0].sample_aspect_ratio == Fraction(3, 2)
            expected = [f.time for f in source.decode(video=0)]
        assert expected[0] == 5.0
        expected = [value - expected[0] for value in expected]
        assert np.allclose([r["timestamp_seconds"] for r in rows], expected[::2], atol=0.001)
        with av.open(str(paths["annotated"])) as video:
            assert video.streams.video[0].sample_aspect_ratio == Fraction(2, 3)
            frames = list(video.decode(video=0))
            assert len(frames) == 6 and frames[0].rotation == 0
            assert (frames[0].width, frames[0].height) == (480, 640)
            assert np.allclose([f.time for f in frames], expected, atol=0.001)
        assert rows[0]["shape"] == [640, 480]
        print(
            json.dumps({"check": "rotated-anamorphic-offset-vfr-video", "passed": True}), flush=True
        )
        for rotation, hflip, vflip in [(0, True, False), (90, False, True)]:
            mirrored = fixtures / f"mirrored-{rotation}.mp4"
            with av.open(str(clip)) as source, av.open(str(mirrored), "w") as out:
                stream = out.add_stream_from_template(source.streams.video[0])
                stream.set_display_rotation(rotation, hflip=hflip, vflip=vflip)
                for packet in source.demux(video=0):
                    if packet.dts is not None:
                        packet.stream = stream
                        out.mux(packet)
            result = run({"source": str(mirrored)})
            assert result["status"] == "failed" and "Mirrored video" in result["error"], result
        print(json.dumps({"check": "mirrored-video-rejected", "passed": True}), flush=True)
        result = run({"source": str(exif_path), "model": str(custom)})
        assert result["status"] == "completed" and result["task"] == "detect", result
        assert result["preview"]["shape"] == [image.width, image.height]
        print(json.dumps({"check": "exif-and-custom-checkpoint", "passed": True}), flush=True)
        custom_yoloe = fixtures / "custom-vocabulary.pt"
        shutil.copy2(model_dir / "yoloe-26s-seg.pt", custom_yoloe)
        result = run(
            {
                "source": str(ASSETS / "bus.jpg"),
                "model": str(custom_yoloe),
                "text_prompts": ["bus", "person"],
            }
        )
        assert result["status"] == "completed" and result["preview"]["observations"], result
        print(json.dumps({"check": "renamed-custom-yoloe-checkpoint", "passed": True}), flush=True)
        result = run({"source": str(corrupt)})
        assert result["status"] == "failed" and "UnidentifiedImageError" in result["error"], result
        print(json.dumps({"check": "corrupt-image", "passed": True}), flush=True)
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
