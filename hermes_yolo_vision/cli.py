"""Standalone diagnostic runner using the same validation and worker as Hermes."""

import argparse
import json
import os
from pathlib import Path

from .jobs import Jobs
from .plugin import readiness
from .requests import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["info", "predict", "track", "job"])
    parser.add_argument("--args", default="{}", help="Tool arguments as JSON")
    parser.add_argument(
        "--home", type=Path, default=Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    )
    parser.add_argument("--wait", type=float, default=900)
    options = parser.parse_args()
    tool = f"yolo_{options.operation}"
    try:
        args = validate(tool, json.loads(options.args), Path.cwd())
        if tool == "yolo_info" and "model" not in args:
            result = readiness()
        else:
            jobs = Jobs(
                options.home / "plugin-data/hermes-yolo-vision",
                options.home / "cache/generated/hermes-yolo-vision",
            )
            try:
                if tool == "yolo_job":
                    result = (
                        jobs.cancel(args["job_id"], "standalone")
                        if args.get("action") == "cancel"
                        else jobs.get(args["job_id"], "standalone", args.get("wait_seconds", 0))
                    )
                else:
                    result = jobs.submit(tool, args, "standalone")
                    if result["status"] == "running":
                        result = jobs.get(result["job_id"], "standalone", max(0, options.wait))
            finally:
                jobs.close()
            if result["status"] == "running":
                result = jobs.get(result["job_id"], "standalone")
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in ("completed", "ready") else 1
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
