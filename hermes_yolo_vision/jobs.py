"""One cancellable subprocess per profile, with durable, inspectable job records."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


class Jobs:
    def __init__(self, data_dir, media_dir, python=sys.executable):
        self.data = Path(data_dir)
        self.media = Path(media_dir)
        self.python = str(python)
        self.records = self.data / "jobs"
        self.records.mkdir(parents=True, exist_ok=True)
        self.media.mkdir(parents=True, exist_ok=True)
        self._closed = threading.Event()
        self._threads = []
        self._mutex = threading.Lock()

    def _lock(self):
        handle = (self.data / "inference.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        return handle

    def _recover(self):
        for path in self.records.glob("*/request.json"):
            record = read_json(path)
            if not (path.parent / "result.json").exists():
                write_json(
                    path.parent / "result.json",
                    {
                        "status": "interrupted",
                        "error": "The worker ended without a result; submit again explicitly.",
                        "job_id": record["job_id"],
                    },
                )

    def submit(
        self,
        operation,
        arguments,
        owner,
        *,
        conversation="",
        session="",
        notify=None,
        max_seconds=900,
        python=None,
    ):
        with self._mutex:
            if self._closed.is_set():
                raise ValueError("Plugin is unloading")
            lock = self._lock()
            if lock is None:
                return {
                    "status": "busy",
                    "message": "One inference job is already active in this profile. Wait or cancel your current job.",
                }
            try:
                self._recover()
                job_id = uuid.uuid4().hex
                directory = self.records / job_id
                directory.mkdir()
                output = self.media / job_id
                output.mkdir()
                record = {
                    "job_id": job_id,
                    "operation": operation,
                    "arguments": arguments,
                    "owner": owner,
                    "conversation": conversation,
                    "session": session,
                    "output": str(output.resolve()),
                    "data": str(self.data.resolve()),
                    "created_at": time.time(),
                    "parent_pid": os.getpid(),
                }
                request = directory / "request.json"
                write_json(request, record)
                env = {**os.environ, "YOLO_AUTOINSTALL": "false", "MPLBACKEND": "Agg"}
                root = str(Path(__file__).resolve().parent.parent)
                env["PYTHONPATH"] = root
                with (directory / "worker.log").open("w") as log:
                    process = subprocess.Popen(
                        [str(python or self.python), "-m", "hermes_yolo_vision.worker", str(request.resolve())],
                        stdout=log,
                        stderr=log,
                        env=env,
                        pass_fds=(lock.fileno(),),
                        start_new_session=True,
                    )
                thread = threading.Thread(
                    target=self._watch,
                    args=(process, directory, record, notify, max_seconds, lock),
                    daemon=True,
                )
                self._threads = [t for t in self._threads if t.is_alive()]
                self._threads.append(thread)
                thread.start()
                lock = None  # The monitor and child retain the admission lock through finalization.
            except Exception:
                if "directory" in locals():
                    write_json(
                        directory / "result.json",
                        {"status": "failed", "error": "Could not start inference worker"},
                    )
                raise
            finally:
                if lock:
                    lock.close()
        return self.get(job_id, owner)

    def _watch(self, process, directory, record, notify, max_seconds, lock):
        try:
            self._finish(process, directory, record, notify, max_seconds)
        finally:
            lock.close()

    def _finish(self, process, directory, record, notify, max_seconds):
        reason = None
        started = time.monotonic()
        while process.poll() is None:
            if self._closed.is_set():
                reason = "interrupted"
            elif (directory / "cancel").exists():
                reason = "cancelled"
            elif time.monotonic() - started > max_seconds:
                reason = "timed_out"
            if reason:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.1)
        payload = directory / "payload.json"
        if reason:
            result = {"status": "failed" if reason == "timed_out" else reason, "error": reason}
        elif process.returncode == 0 and payload.exists():
            try:
                result = read_json(payload)
            except (OSError, ValueError):
                result = {"status": "failed", "error": "Worker produced an invalid result"}
        else:
            result = {
                "status": "failed",
                "error": "Inference worker exited unexpectedly; inspect worker.log in the job directory.",
            }
        result.update(job_id=record["job_id"], elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(directory / "result.json", result)
        if (
            notify
            and result["status"] in ("completed", "failed")
            and not (directory / "suppress_notification").exists()
        ):
            try:
                accepted = bool(notify(record["job_id"]))
            except Exception:
                accepted = False
            write_json(
                directory / "notification.json", {"accepted": accepted, "delivery_verified": False}
            )

    def get(self, job_id, owner, wait_seconds=0):
        directory = self.records / job_id
        request = directory / "request.json"
        if not request.exists() or read_json(request)["owner"] != owner:
            raise ValueError("Job not found in this conversation route")
        deadline = time.monotonic() + wait_seconds
        while not (directory / "result.json").exists():
            lock = self._lock()
            if lock:
                # The monitor may be finishing after the worker released its inherited lock.
                try:
                    if not any(t.is_alive() for t in self._threads):
                        self._recover()
                finally:
                    lock.close()
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if (directory / "result.json").exists():
            result = read_json(directory / "result.json")
            if (directory / "notification.json").exists():
                result["notification"] = read_json(directory / "notification.json")
            return result
        progress = directory / "progress.json"
        return {
            "status": "running",
            "job_id": job_id,
            "progress": read_json(progress) if progress.exists() else {"phase": "starting"},
            "next_step": "Call yolo_job with this job_id and wait_seconds=30; do not resubmit the same request.",
        }

    def cancel(self, job_id, owner):
        current = self.get(job_id, owner)
        if current["status"] in TERMINAL:
            return current
        (self.records / job_id / "cancel").touch()
        result = self.get(job_id, owner, 5)
        if result["status"] == "running":
            result["cancel_requested"] = True
        return result

    def suppress(self, session_id):
        for path in self.records.glob("*/request.json"):
            record = read_json(path)
            if session_id in (record["conversation"], record["session"], record.get("current_session")):
                (path.parent / "suppress_notification").touch()

    def compact(self, old_session_id, session_id):
        with self._mutex:
            for path in self.records.glob("*/request.json"):
                record = read_json(path)
                if old_session_id == record.get("current_session", record["session"]):
                    record["current_session"] = session_id
                    write_json(path, record)

    def close(self):
        with self._mutex:
            self._closed.set()
        for thread in self._threads:
            thread.join(timeout=5)
