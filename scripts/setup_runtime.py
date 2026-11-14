"""Install the inference runtime explicitly, without modifying Hermes dependencies."""

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-backend", default="cpu", help="uv PyTorch backend, e.g. cpu or auto")
    options = parser.parse_args()
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    root = Path(__file__).resolve().parent.parent
    runtime = root / ".venv"
    if not (runtime / "bin/python").exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(runtime)], check=True, cwd=root)
    subprocess.run([uv, "pip", "install", "--python", str(runtime / "bin/python"), "--torch-backend", options.torch_backend, str(root)], check=True, cwd=root)
    subprocess.run([str(runtime / "bin/python"), "-I", "-m", "hermes_yolo_vision.cli", "info"], check=True)


if __name__ == "__main__":
    main()
