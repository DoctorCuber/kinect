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


_splash = None
_splash_label = None


def show_splash() -> None:
    """A small window that appears while the heavier libraries load."""
    global _splash, _splash_label
    if _splash is not None:
        return
    import tkinter as tk

    window = tk.Tk()
    window.title("Opening Kinect")
    window.geometry("440x150")
    window.resizable(False, False)
    window.attributes("-topmost", True)
    window.configure(bg="#E4EEF8")
    label = tk.Label(
        window,
        text="Opening the Kinect window.\nThis can take a minute.",
        font=("Segoe UI", 13),
        justify="center",
        bg="#E4EEF8",
        fg="#1D1D1F",
    )
    label.pack(expand=True, fill="both", padx=24, pady=24)
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = max(0, (window.winfo_screenwidth() - width) // 2)
    y = max(0, (window.winfo_screenheight() - height) // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.update()
    _splash = window
    _splash_label = label


def splash_status(text: str) -> None:
    if _splash is None or _splash_label is None:
        return
    _splash_label.configure(text=text)
    _splash.update()


def close_splash() -> None:
    global _splash, _splash_label
    if _splash is None:
        return
    try:
        _splash.destroy()
    except Exception:
        pass
    _splash = None
    _splash_label = None


def _is_admin() -> bool:
    if os.name != "nt":
        return True
    import ctypes

    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def _relaunch_as_admin() -> None:
    """Show the Windows approval prompt, then start again with administrator rights."""
    import ctypes

    executable = str(sys.executable)
    pythonw = VENV_PYTHON.with_name("pythonw.exe")
    if pythonw.exists() and _same_interpreter(VENV_PYTHON):
        executable = str(pythonw)
    script = str(Path(sys.argv[0]).resolve())
    params = subprocess.list2cmdline([script, *sys.argv[1:]])
    shell = ctypes.windll.shell32
    shell.ShellExecuteW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_int,
    ]
    shell.ShellExecuteW.restype = ctypes.c_void_p
    result = shell.ShellExecuteW(None, "runas", executable, params, str(ROOT), 1)
    code = result if isinstance(result, int) else 0
    if code <= 32:
        raise SystemExit("Administrator approval was cancelled.")
    raise SystemExit(0)


def ensure() -> None:
    """Install into .venv when needed, then start the window in its own process."""
    if not _is_admin():
        _relaunch_as_admin()
    if os.environ.get("KINECT_SETUP_CHILD") == "1" or _ready():
        show_splash()
        return
    if not VENV_PYTHON.exists():
        print("Creating the Python environment in .venv", flush=True)
        _create_venv()
    if not _stamp_ok():
        print("Installing packages. The first run downloads PyTorch and can take several minutes.", flush=True)
        _install(VENV_PYTHON)
    if _same_interpreter(VENV_PYTHON):
        show_splash()
        return
    print("Starting the Kinect window.", flush=True)
    _handoff()


def _ready() -> bool:
    return VENV_PYTHON.exists() and _same_interpreter(VENV_PYTHON) and _stamp_ok()


def _same_interpreter(python: Path) -> bool:
    """True when this process is the project environment, including pythonw.exe."""
    try:
        current = Path(sys.executable).resolve()
        target = python.resolve()
    except OSError:
        return False
    if current == target:
        return True
    return current.parent == target.parent and current.stem.lower() in {"python", "pythonw"} and target.stem.lower() in {"python", "pythonw"}


def _handoff() -> None:
    """Leave the console behind and open the window as its own program."""
    pythonw = VENV_PYTHON.with_name("pythonw.exe")
    launcher = pythonw if pythonw.exists() else VENV_PYTHON
    environment = os.environ.copy()
    environment["KINECT_SETUP_CHILD"] = "1"
    subprocess.Popen(
        [str(launcher), *sys.argv],
        env=environment,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    raise SystemExit(0)


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
