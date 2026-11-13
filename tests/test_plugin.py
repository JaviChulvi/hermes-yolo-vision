import json
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

from hermes_yolo_vision.plugin import register


class Context:
    def __init__(self, cwd):
        self.cwd = cwd
        self.tools, self.hooks, self.notifications = {}, {}, []
        self.config = {"python": sys.executable}

    def get_config(self, key, default):
        return self.config.get(key, default)

    def session_cwd(self):
        return str(self.cwd)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]

    def register_hook(self, name, handler):
        self.hooks[name] = handler

    def register_skill(self, *args):
        pass

    def on_unload(self, handler):
        self.close = handler

    def inject_message(self, content, **kwargs):
        self.notifications.append((content, kwargs))
        return True


def context(tmp_path, monkeypatch):
    modules = {
        "hermes_constants": {"get_hermes_home": lambda: tmp_path},
        "plugins.plugin_storage": {"plugin_data_dir": lambda _: tmp_path / "data"},
        "gateway.session_context": {"get_session_env": lambda *a: "telegram:private:test"},
        "agent.file_safety": {"get_read_block_error": lambda _: None},
    }
    for name, values in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    ctx = Context(tmp_path)
    (tmp_path / "photo.jpg").write_bytes(b"test")
    register(ctx)
    return ctx


def test_inline_completion_does_not_inject_duplicate_turn(tmp_path, monkeypatch, worker):
    ctx = context(tmp_path, monkeypatch)
    try:
        result = json.loads(ctx.tools["yolo_predict"]({"source": "photo.jpg"}, session_id="old"))
        assert result["status"] == "completed"
    finally:
        ctx.close()
    assert ctx.notifications == []


def test_native_reset_signature_suppresses_old_session(tmp_path, monkeypatch, worker):
    ctx = context(tmp_path, monkeypatch)
    try:
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(ctx.tools["yolo_predict"], {"source": "photo.jpg"}, session_id="old")
            deadline = time.monotonic() + 3
            while not list((tmp_path / "data/jobs").glob("*/request.json")) and time.monotonic() < deadline:
                time.sleep(0.01)
            ctx.hooks["post_compaction"](session_id="compacted", old_session_id="old")
            ctx.hooks["post_compaction"](session_id="compacted-again", old_session_id="compacted")
            ctx.hooks["on_session_reset"](session_id="new", old_session_id="compacted-again")
            result = json.loads(future.result(timeout=5))
            assert (tmp_path / "data/jobs" / result["job_id"] / "suppress_notification").exists()
    finally:
        ctx.close()
    assert ctx.notifications == []


def test_runtime_setting_is_resolved_for_each_job(tmp_path, monkeypatch, worker):
    ctx = context(tmp_path, monkeypatch)
    import subprocess

    launch = subprocess.Popen
    executables = []

    def record(command, **kwargs):
        executables.append(command[0])
        return launch(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)
    try:
        for python in ("/first/python", "/second/python"):
            ctx.config["python"] = python
            result = json.loads(ctx.tools["yolo_predict"]({"source": "photo.jpg"}, session_id="old"))
            assert result["status"] == "completed"
            time.sleep(0.15)
    finally:
        ctx.close()
    assert executables == ["/first/python", "/second/python"]
