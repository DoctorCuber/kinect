"""Open the detector and show a message if startup fails."""

import traceback
from pathlib import Path


def main() -> int:
    try:
        from setup_runtime import ensure

        ensure()
        from kinect_app import run

        return run("detect")
    except SystemExit as exc:
        if exc.code not in (0, None):
            _show_error(str(exc))
        raise
    except Exception:
        _show_error(traceback.format_exc())
        return 1


def _show_error(text: str) -> None:
    log = Path(__file__).resolve().parent / "startup_error.log"
    log.write_text(text, encoding="utf-8")
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Kinect", "The window could not open.\n\n" + text[-1200:])
    root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
