import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_yolo_vision.jobs import Jobs, write_json


def test_atomic_admission_is_shared_across_managers(tmp_path, worker):
    a = Jobs(tmp_path / "data", tmp_path / "media")
    b = Jobs(tmp_path / "data", tmp_path / "media")
    barrier = threading.Barrier(2)

    def submit(jobs):
        barrier.wait()
        return jobs.submit("test", {"delay": 1}, "one")

    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(submit, [a, b]))
        assert sorted(r["status"] for r in results) == ["busy", "running"]
        job = next(r["job_id"] for r in results if r["status"] == "running")
        assert a.get(job, "one", 5)["status"] == "completed"
        with pytest.raises(ValueError, match="not found"):
            b.get(job, "another-chat")
    finally:
        a.close()
        b.close()


def test_cancel_timeout_and_unload_stop_owned_workers(tmp_path, worker):
    jobs = Jobs(tmp_path / "data", tmp_path / "media")
    job = jobs.submit("test", {"delay": 20}, "one")["job_id"]
    assert jobs.cancel(job, "one")["status"] == "cancelled"
    time.sleep(0.2)
    job = jobs.submit("test", {"delay": 20}, "one", max_seconds=0.1)["job_id"]
    assert jobs.get(job, "one", 5)["error"] == "timed_out"
    time.sleep(0.2)
    job = jobs.submit("test", {"delay": 20}, "one")["job_id"]
    jobs.close()
    assert jobs.get(job, "one")["status"] == "interrupted"


def test_reset_suppresses_notification_and_recovery_never_signals_pid(tmp_path, worker):
    notifications = []
    jobs = Jobs(tmp_path / "data", tmp_path / "media")
    try:
        job = jobs.submit(
            "test",
            {"delay": 0.5},
            "chat",
            session="segment",
            conversation="root",
            notify=notifications.append,
        )["job_id"]
        jobs.suppress("root")
        assert jobs.get(job, "chat", 5)["status"] == "completed"
        assert notifications == []
        directory = jobs.records / ("a" * 32)
        directory.mkdir()
        write_json(directory / "request.json", {"job_id": "a" * 32, "owner": "chat", "pid": 1})
        assert jobs.get("a" * 32, "chat", 2)["status"] == "interrupted"
    finally:
        jobs.close()
