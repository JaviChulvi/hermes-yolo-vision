"""Hermes integration: public registration, session context, and media delivery."""

import importlib.metadata
import json
import subprocess
import sys
import threading
from pathlib import Path

from .jobs import Jobs
from .requests import validate
from .schemas import DEFAULT_MODELS, SCHEMAS, TEXT_MODEL


def readiness(python=None):
    if python and str(python) != sys.executable:
        try:
            probe = subprocess.run(
                [str(python), "-c", "import sys; sys.path.insert(0, sys.argv[1]); from hermes_yolo_vision.plugin import readiness; import json; print(json.dumps(readiness()))", str(Path(__file__).resolve().parent.parent)],
                capture_output=True, text=True, timeout=10,
            )
            return json.loads(probe.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {"status": "missing_dependencies", "python": str(python), "error": "Run scripts/setup_runtime.py in the plugin directory before using vision tools."}
    packages = {}
    for name in ("ultralytics", "torch", "av", "clip", "lap"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "status": "ready" if all(packages.values()) else "missing_dependencies",
        "packages": packages,
        "defaults": DEFAULT_MODELS,
        "text_model": TEXT_MODEL,
        "device": "cpu",
        "inputs": "local image/video files and downloaded Hermes attachments",
        "preparation": "First use downloads the selected weights; YOLOE also downloads a text encoder. No packages are installed by a handler.",
    }


def register(ctx):
    managers = {}
    mutex = threading.Lock()

    def runtime():
        return ctx.get_config("python", "") or str(Path(__file__).resolve().parent.parent / ".venv/bin/python")

    def manager():
        from hermes_constants import get_hermes_home
        from plugins.plugin_storage import plugin_data_dir

        data = plugin_data_dir("hermes-yolo-vision")
        with mutex:
            if data not in managers:
                managers[data] = Jobs(
                    data, Path(get_hermes_home()) / "cache/generated/hermes-yolo-vision"
                )
            return managers[data]

    def handle(tool, args, **kwargs):
        try:
            from agent.file_safety import get_read_block_error
            from gateway.session_context import get_session_env

            params = validate(
                tool, args, ctx.session_cwd(), get_read_block_error, ctx.get_config("device", "cpu")
            )
            if tool == "yolo_info" and "model" not in params:
                return json.dumps(readiness(runtime()))
            route = get_session_env("HERMES_SESSION_KEY", "")
            conversation = kwargs.get("conversation_id") or kwargs.get("session_id", "")
            owner = route or conversation
            if not owner:
                raise ValueError("Use this tool from a Hermes conversation with session context")
            jobs = manager()
            if tool == "yolo_job":
                if params.get("action") == "cancel":
                    result = jobs.cancel(params["job_id"], owner)
                else:
                    result = jobs.get(params["job_id"], owner, params.get("wait_seconds", 0))
            else:
                response_ready = threading.Event()
                background = False

                def notify(job_id):
                    response_ready.wait()
                    return (
                        ctx.inject_message(
                            f"YOLO job {job_id} finished. Call yolo_job to retrieve its actual result and deliver requested artifacts. This notification alone does not indicate success.",
                            session_key=route,
                        )
                        if route and background and not (jobs.records / job_id / "suppress_notification").exists()
                        else False
                    )

                try:
                    result = jobs.submit(
                        tool,
                        params,
                        owner,
                        conversation=conversation,
                        session=kwargs.get("session_id", ""),
                        notify=notify,
                        max_seconds=max(1, int(ctx.get_config("max_job_seconds", 900))),
                        python=runtime(),
                    )
                    if result["status"] == "running":
                        result = jobs.get(result["job_id"], owner, 20)
                    background = result["status"] == "running"
                finally:
                    response_ready.set()
            return json.dumps(result, allow_nan=False)
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)})

    for spec in SCHEMAS:

        def bound(args, _tool=spec["name"], **kwargs):
            return handle(_tool, args, **kwargs)

        ctx.register_tool(name=spec["name"], toolset="yolo_vision", schema=spec, handler=bound)

    def reset(session_id=None, old_session_id=None, **kwargs):
        previous = old_session_id or session_id
        if previous:
            for jobs in list(managers.values()):
                jobs.suppress(previous)

    def close():
        for jobs in list(managers.values()):
            jobs.close()

    def compact(session_id=None, old_session_id=None, **kwargs):
        if session_id and old_session_id:
            for jobs in list(managers.values()):
                jobs.compact(old_session_id, session_id)

    ctx.register_hook("on_session_reset", reset)
    ctx.register_hook("on_session_delete", reset)
    ctx.register_hook("post_compaction", compact)
    ctx.on_unload(close)
    ctx.register_skill(
        "yolo-vision", Path(__file__).resolve().parent / "skills/yolo-vision/SKILL.md"
    )
