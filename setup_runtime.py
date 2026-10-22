"""Create .venv and install packages the first time a Kinect script runs.

The Kinect USB driver is not a Python package. Windows has to already have
Kinect for Windows SDK 2.0 installed. Python itself has to be installed too.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
STAMP = VENV_DIR / ".kinect-setup"
REQUIREMENTS = ROOT / "requirements.txt"
TORCH_PACKAGES = ("torch==2.14.1+cu130", "torchvision==0.29.1+cu130")
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def ensure() -> None:
    """Install into .venv when needed, then restart the script with that Python."""
    if os.environ.get("KINECT_SETUP_CHILD") == "1":
        return
    if _ready():
        return
    if not VENV_PYTHON.exists():
        print("Creating the Python environment in .venv", flush=True)
        _create_venv()
    if not _stamp_ok():
        print("Installing packages. The first run downloads PyTorch and can take several minutes.", flush=True)
        _install(VENV_PYTHON)
    if _same_interpreter(VENV_PYTHON):
        return
    print("Starting the Kinect window.", flush=True)
    environment = os.environ.copy()
    environment["KINECT_SETUP_CHILD"] = "1"
    raise SystemExit(subprocess.call([str(VENV_PYTHON), *sys.argv], env=environment))


def _ready() -> bool:
    return VENV_PYTHON.exists() and _same_interpreter(VENV_PYTHON) and _stamp_ok()


def _same_interpreter(python: Path) -> bool:
    try:
        return Path(sys.executable).resolve() == python.resolve()
    except OSError:
        return False


def _stamp_text() -> str:
    requirements = REQUIREMENTS.read_text(encoding="utf-8") if REQUIREMENTS.exists() else ""
    payload = requirements + "\n" + " ".join(TORCH_PACKAGES) + "\n" + TORCH_INDEX
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stamp_ok() -> bool:
    try:
        return STAMP.read_text(encoding="utf-8").strip() == _stamp_text()
    except OSError:
        return False


def _create_venv() -> None:
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Could not create .venv. Install Python 3.12 from python.org and check Add python.exe to PATH."
        ) from exc
    if not VENV_PYTHON.exists():
        raise SystemExit("The environment was created, but .venv\\Scripts\\python.exe is missing.")


def _install(python: Path) -> None:
    subprocess.call([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    try:
        subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        subprocess.check_call([str(python), "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", TORCH_INDEX])
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Package install failed. Check the network connection, then run the program again.") from exc
    STAMP.write_text(_stamp_text() + "\n", encoding="utf-8")


if __name__ == "__main__":
    ensure()
