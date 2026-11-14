import subprocess
import sys

import pytest


@pytest.fixture
def worker(monkeypatch):
    popen = subprocess.Popen

    def launch(command, **kwargs):
        code = """
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
r = json.loads(p.read_text())
time.sleep(r['arguments'].get('delay', 0.2))
(p.parent / 'payload.json').write_text(json.dumps({'status':'completed', 'value':42}))
"""
        return popen([sys.executable, "-c", code, command[-1]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
