import json
import sys
from pathlib import Path

import jsonschema
import pytest

from hermes_yolo_vision.plugin import register
from hermes_yolo_vision.requests import validate
from hermes_yolo_vision.schemas import SCHEMAS


def test_registration_is_lightweight_and_matches_manifest():
    import subprocess

    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hermes_yolo_vision.plugin import register; import sys; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules",
        ],
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr

    class Context:
        def __init__(self):
            self.tools = {}
            self.hooks = []

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_hook(self, name, handler):
            self.hooks.append(name)

        def register_skill(self, name, path):
            assert name == "yolo-vision" and path.is_file()

        def on_unload(self, handler):
            self.close = handler

    ctx = Context()
    register(ctx)
    assert set(ctx.tools) == {s["name"] for s in SCHEMAS}
    assert set(ctx.hooks) == {"on_session_reset", "on_session_delete", "post_compaction"}
    manifest = Path("plugin.yaml").read_text()
    for spec in SCHEMAS:
        assert f"  - {spec['name']}" in manifest
        jsonschema.Draft202012Validator.check_schema(spec["parameters"])
    ctx.close()


@pytest.mark.parametrize(
    "args",
    [
        {"source": "https://example.org/image.jpg"},
        {"source": "frame.jpg", "vid_stride": True},
        {"source": "frame.jpg", "conf": float("nan")},
        {"source": "frame.jpg", "save_raw": "true"},
        {"source": "frame.jpg", "device": "cpu; echo bad"},
        {"source": "frame.jpg", "text_prompts": ["helmet"], "task": "depth"},
        {"source": "frame.jpg", "text_prompts": ["helmet"], "classes": ["person"]},
        {"source": "frame.jpg", "classes": [""]},
        {"source": "frame.jpg", "unknown": 1},
    ],
)
def test_bad_inputs_fail_before_worker(tmp_path, args):
    (tmp_path / "frame.jpg").write_bytes(b"fixture")
    with pytest.raises(ValueError):
        validate("yolo_predict", args, tmp_path)


def test_model_defaults_and_file_policy(tmp_path):
    source = tmp_path / "frame.jpg"
    source.write_bytes(b"fixture")
    result = validate("yolo_predict", {"source": "frame.jpg", "text_prompts": ["helmet"]}, tmp_path)
    assert result["source"] == str(source)
    assert result["model"] == "yoloe-26s-seg.pt" and result["task"] == "segment"
    assert result["vid_stride"] == 1 and result["device"] == "cpu"
    with pytest.raises(ValueError, match="policy denied"):
        validate("yolo_predict", {"source": "frame.jpg"}, tmp_path, lambda _: "policy denied")
    with pytest.raises(ValueError, match="video"):
        validate("yolo_track", {"source": "frame.jpg"}, tmp_path)
    with pytest.raises(ValueError, match="job_id"):
        validate("yolo_job", {"job_id": "../../escape"}, tmp_path)


def test_inference_schema_accepts_documented_request():
    spec = next(s for s in SCHEMAS if s["name"] == "yolo_predict")
    jsonschema.validate(json.loads('{"source":"image.png","task":"pose"}'), spec["parameters"])
