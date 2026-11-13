"""Run matched Hermes conversations on an authorized host; retain ungraded evidence."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def bootstrap(repo):
    sys.path.insert(0, str(repo))
    __import__("hermes_bootstrap")


def worker(options):
    bootstrap(options.hermes)
    home, work = options.run / "home", options.run / "work"
    os.environ.update(
        HERMES_HOME=str(home),
        TERMINAL_CWD=str(work),
        YOLO_AUTOINSTALL="false",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    os.environ["PATH"] = str(options.runtime.parent) + os.pathsep + os.environ.get("PATH", "")
    os.chdir(work)
    spec = json.loads((options.run / "request.json").read_text())
    from hermes_cli.plugins import get_plugin_manager
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    runtime = resolve_runtime_provider(
        requested=spec["model"]["provider"], target_model=spec["model"]["default"]
    )
    start = time.monotonic()
    calls = []
    agent = None

    def on_tool(tool_id, name, args):
        calls.append({"name": name, "arguments": args, "elapsed_seconds": time.monotonic() - start})
        write(options.run / "progress.json", {"calls": len(calls), "last_tool": name, "tool_calls": calls, "start_monotonic": start})
        if len(calls) > 20 and agent is not None:
            agent.interrupt("Evaluation tool call budget reached")

    system = (
        "Work only in the provided workspace and generated-media directory. Do not inspect evaluation "
        "scripts, rubric files, other runs, or other agents. The active Python has Ultralytics, PyAV, "
        "and its dependencies installed. Small official model weights are already in the current directory. "
        "Do not install packages or use a different provider. Complete the user request using available tools. "
        "For reproducibility, save handoff.json in this workspace with an artifacts list of objects "
        "{kind,path}, where kind is annotated, observations, metadata, or raw_arrays. Use absolute paths. "
        "List only files actually created for the user; do not invent outputs. This manifest supplements "
        "the normal user-facing answer and file links. If input is missing or invalid, say so. "
    )
    system += f"The authorized interpreter is {options.runtime}. Invoke it with -I so Hermes's terminal environment cannot replace its dependencies. "
    if options.arm in ("stock", "skill"):
        system += "Use ordinary Hermes tools and the Ultralytics Python/CLI interfaces; the native YOLO plugin is not part of this environment. "
    if options.arm == "skill":
        system += "\n" + (options.plugin / "evals/baseline-skill/SKILL.md").read_text()
    elif options.arm == "plugin":
        system += (
            "\n" + (options.plugin / "hermes_yolo_vision/skills/yolo-vision/SKILL.md").read_text()
        )
    toolsets = ["terminal", "file", "vision", "skills"] + (
        ["yolo_vision"] if options.arm == "plugin" else []
    )
    turns = []
    try:
        agent = AIAgent(
            model=spec["model"]["default"],
            provider=runtime.get("provider"),
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            api_mode=runtime.get("api_mode"),
            credential_pool=runtime.get("credential_pool"),
            max_iterations=21,
            run_budget_seconds=300,
            enabled_toolsets=toolsets,
            session_id="yolo-eval-" + uuid.uuid4().hex,
            skip_memory=True,
            skip_background_review=True,
            skip_context_files=True,
            quiet_mode=True,
            cwd=str(work),
            platform="cli",
            reasoning_config={"effort": "low"},
            ephemeral_system_prompt=system,
            tool_start_callback=on_tool,
        )
        if options.arm == "plugin":
            from tools.registry import registry

            assert registry.get_entry("yolo_predict"), "Native plugin was not registered"
        from model_tools import handle_function_call

        probe_code = "import sys,ultralytics,av,clip,lap,json; print(json.dumps({'python':sys.executable,'ultralytics':ultralytics.__version__,'av':av.__version__}))"
        probe = handle_function_call(
            "terminal",
            {
                "command": f"{shlex.quote(str(options.runtime))} -I -c {shlex.quote(probe_code)}",
                "timeout": 30,
            },
            session_id=agent.session_id,
            task_id=agent.session_id,
        )
        write(options.run / "terminal-preflight.json", {"result": probe})
        assert "8.4.212" in str(probe) and str(options.runtime) in str(probe), (
            "Actual terminal runtime preflight failed"
        )
        start = time.monotonic()
        prompt = spec["case"]["prompt"]
        if spec["attachment"]:
            prompt += "\n\n[Attachment downloaded by the host: " + spec["attachment"] + "]"
            if spec["case"].get("attachment_mode") != "file" and Path(spec["attachment"]).suffix.lower() in (".jpg", ".jpeg", ".png"):
                from agent.image_routing import build_native_content_parts, decide_image_input_mode
                from hermes_cli.config import load_config

                mode = decide_image_input_mode(
                    runtime.get("provider", ""), spec["model"]["default"], load_config()
                )
                if mode == "native":
                    parts, _ = build_native_content_parts(prompt, [spec["attachment"]])
                    if any(part.get("type") == "image_url" for part in parts):
                        prompt = parts
        first = agent.run_conversation(prompt)
        turns.append(first)
        followup = spec["case"].get("followup")
        if (
            followup
            and first.get("completed")
            and len(calls) < 20
            and time.monotonic() - start < 300
        ):
            agent.run_budget_seconds = max(1, 300 - (time.monotonic() - start))
            turns.append(agent.run_conversation(followup, conversation_history=first["messages"]))
        result = {
            "status": "finished",
            "turns": turns,
            "tool_calls": calls,
            "elapsed_seconds": time.monotonic() - start,
        }
    except BaseException as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "turns": turns,
            "tool_calls": calls,
            "elapsed_seconds": time.monotonic() - start,
        }
    finally:
        get_plugin_manager().unload()
    result.update(
        arm=options.arm,
        case_id=spec["case"]["id"],
        repeat=options.repeat,
        request_sha256=hashlib.sha256((options.run / "request.json").read_bytes()).hexdigest(),
        grading="pending",
    )
    try:
        write(options.run / "result.json", result)
    finally:
        if agent is not None:
            agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("hermes", "runtime", "plugin", "fixtures", "output", "auth-home", "models"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--phase", choices=["development", "heldout"], default="development")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["stock", "skill", "plugin"],
        default=["stock", "skill", "plugin"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--repeat", type=int)
    opts = parser.parse_args()
    opts.runtime = opts.runtime.absolute()
    for key in ("hermes", "plugin", "fixtures", "output", "auth_home", "models"):
        setattr(opts, key, getattr(opts, key).resolve())
    if opts.worker:
        signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Runner timeout"))
        )
        worker(opts)
        return
    bootstrap(opts.hermes)
    import hermes_yaml as yaml

    cfg = yaml.safe_load((opts.auth_home / "config.yaml").read_text())
    model = {
        key: cfg["model"][key] for key in ("default", "provider", "base_url") if key in cfg["model"]
    }
    manifest = json.loads((opts.fixtures / "manifest.json").read_text())
    for entry in manifest["files"].values():
        assert (
            hashlib.sha256((opts.fixtures / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    cases = manifest["development_cases" if opts.phase == "development" else "cases"]
    if opts.cases:
        cases = [c for c in cases if c["id"] in opts.cases]
        assert len(cases) == len(set(opts.cases)), "Unknown case id"
    opts.output.mkdir(parents=True, exist_ok=True)
    versions = subprocess.check_output(
        [
            str(opts.runtime),
            "-I",
            "-c",
            "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version, 'packages':{n:importlib.metadata.version(n) for n in ['ultralytics','torch','torchvision','av','lap','clip','numpy','pillow']}}))",
        ],
        text=True,
    )
    source_paths = [
        *opts.plugin.glob("hermes_yolo_vision/**/*.py"),
        *opts.plugin.glob("hermes_yolo_vision/**/*.md"),
        *opts.plugin.glob("evals/*.py"),
        *opts.plugin.glob("evals/*.json"),
        *opts.plugin.glob("evals/**/*.md"),
        opts.plugin / "plugin.yaml",
        opts.plugin / "pyproject.toml",
    ]
    study = {
        "model": model,
        "reasoning": "low",
        "phase": opts.phase,
        "versions": json.loads(versions),
        "hermes_commit": subprocess.check_output(
            ["git", "-C", str(opts.hermes), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {
            str(p.relative_to(opts.plugin)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_paths)
        },
        "weights": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(opts.models.iterdir())
            if p.suffix in (".pt", ".ts")
        },
        "manifest_sha256": hashlib.sha256(
            (opts.fixtures / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    study_path = opts.output / "study.json"
    if study_path.exists():
        assert json.loads(study_path.read_text()) == study, (
            "Study identity changed; use a new output directory"
        )
    else:
        write(study_path, study)
    mapping_path = opts.output / "runs.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    for repeat in range(opts.repeats):
        for index, case in enumerate(cases):
            offset = (index + repeat) % len(opts.arms)
            order = opts.arms[offset:] + opts.arms[:offset]
            for arm in order:
                key = f"{case['id']}--{repeat}--{arm}"
                if key not in mapping:
                    mapping[key] = uuid.uuid4().hex
                    write(mapping_path, mapping)
                run = opts.output / mapping[key]
                if (run / "result.json").exists():
                    print(json.dumps({"skipped_existing": run.name}), flush=True)
                    continue
                if shutil.disk_usage(opts.output).free < 1024**3:
                    raise SystemExit("Less than 1 GiB free: evaluation stopped, coverage incomplete; preserve evidence before freeing space")
                home, work = run / "home", run / "work"
                home.mkdir(parents=True, exist_ok=True, mode=0o700)
                work.mkdir(exist_ok=True)
                (home / "auth.json").symlink_to(opts.auth_home / "auth.json")
                config = {
                    "model": model,
                    "plugins": {
                        "enabled": ["hermes-yolo-vision"] if arm == "plugin" else [],
                        "entries": {
                            "hermes-yolo-vision": {"settings": {"python": str(opts.runtime)}}
                        },
                    },
                    "terminal": {"backend": "local", "cwd": str(work)},
                    "agent": {"max_turns": 20, "reasoning_effort": "low"},
                    "memory": {"memory_enabled": False, "user_profile_enabled": False},
                }
                (home / "config.yaml").write_text(json.dumps(config))
                (home / "cache/generated").mkdir(parents=True)
                if arm == "plugin":
                    (home / "plugins").mkdir()
                    (home / "plugins/hermes-yolo-vision").symlink_to(
                        opts.plugin, target_is_directory=True
                    )
                data = home / "plugin-data/hermes-yolo-vision"
                data.mkdir(parents=True)
                (data / "models").symlink_to(opts.models, target_is_directory=True)
                for weights in opts.models.iterdir():
                    if weights.suffix in (".pt", ".ts"):
                        (work / weights.name).symlink_to(weights)
                attachment = None
                if case["fixture"]:
                    source = opts.fixtures / manifest["files"][case["fixture"]]["path"]
                    destination = work / ("attachment" + source.suffix)
                    destination.symlink_to(source)
                    attachment = str(destination)
                write(
                    run / "request.json",
                    {
                        "case": case,
                        "attachment": attachment,
                        "model": model,
                        "phase": opts.phase,
                        "manifest_sha256": hashlib.sha256(
                            (opts.fixtures / "manifest.json").read_bytes()
                        ).hexdigest(),
                    },
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--run",
                    str(run),
                    "--arm",
                    arm,
                    "--repeat",
                    str(repeat),
                ]
                for key in (
                    "hermes",
                    "runtime",
                    "plugin",
                    "fixtures",
                    "output",
                    "auth_home",
                    "models",
                ):
                    command += ["--" + key.replace("_", "-"), str(getattr(opts, key))]
                start = time.monotonic()
                with (run / "process.log").open("w") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=log, start_new_session=True
                    )
                    try:
                        process.wait(timeout=330)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if not (run / "result.json").exists():
                    progress_path = run / "progress.json"
                    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                    write(
                        run / "result.json",
                        {
                            "status": "failed",
                            "error": "Child exited or exceeded 330s outer deadline",
                            "elapsed_seconds": time.monotonic() - progress.get("start_monotonic", start),
                            "tool_calls": progress.get("tool_calls", []),
                            "observed_tool_calls": progress.get("calls", 0),
                            "arm": arm,
                            "case_id": case["id"],
                            "repeat": repeat,
                            "grading": "pending",
                        },
                    )
                result = json.loads((run / "result.json").read_text())
                print(
                    json.dumps(
                        {
                            "run": run.name,
                            "status": result["status"],
                            "elapsed_seconds": round(result["elapsed_seconds"], 2),
                            "tool_calls": len(result.get("tool_calls", [])),
                        }
                    ),
                    flush=True,
                )
                text = json.dumps(result).lower()
                if any(
                    term in text
                    for term in (
                        "insufficient_quota",
                        "usage_limit_reached",
                        "invalid_api_key",
                        "authenticationerror",
                    )
                ):
                    raise SystemExit(
                        "Provider access/quota failure: evaluation stopped, coverage incomplete"
                    )


if __name__ == "__main__":
    main()
