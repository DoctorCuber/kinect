"""One window with Detect and Train modes. The Kinect stays open across the switch."""

import customtkinter as ctk

from kinect_view import KinectSensor
from sleepwalk_detector import DetectorApp
from train_person_model import TrainerApp
from setup_runtime import close_splash, splash_status
from ui_kit import SECONDARY, WINDOW, AppButton, apply_theme, body_font, mount_frost, open_settings, saved_theme, segmented, wash_frame


class KinectApp(ctk.CTk):
    def __init__(self, sensor: KinectSensor | None, camera_error: str, mode: str) -> None:
        super().__init__()
        self.title("Kinect")
        self.geometry("1040x860")
        self.minsize(900, 720)
        self.configure(fg_color=WINDOW)
        self.sensor = sensor
        self._closed = False
        self._mode = "Detect" if mode == "detect" else "Train"

        bar = wash_frame(self)
        bar.pack(fill="x", padx=24, pady=(20, 12))
        self.switch = segmented(bar, ["Detect", "Train"], self._switch, height=32)
        self.switch.set(self._mode)
        self.switch.pack(side="left")
        self.notice = ctk.CTkLabel(bar, text="", text_color=SECONDARY, font=body_font(12))
        self.notice.pack(side="left", padx=12)
        AppButton(bar, text="Settings", command=self._open_settings).pack(side="right")
        self.theme_switch = segmented(bar, ["Light", "Dark"], self._theme, height=32)
        self.theme_switch.set("Dark" if saved_theme() == "dark" else "Light")
        self.theme_switch.pack(side="right", padx=(0, 8))

        self.host = wash_frame(self)
        self.host.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self.detect = DetectorApp(self.host, sensor, camera_error)
        self.train = TrainerApp(self.host, sensor, camera_error)
        self._show(self._mode)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Control-comma>", lambda _event: self._open_settings())
        mount_frost(self)
        self.after(40, self._tick)
        self.after(0, self._raise)

    def _raise(self) -> None:
        self.lift()
        self.attributes("-topmost", True)
        self.after(400, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def _show(self, mode: str) -> None:
        self.detect.pack_forget()
        self.train.pack_forget()
        panel = self.detect if mode == "Detect" else self.train
        panel.pack(fill="both", expand=True)

    def _switch(self, value: str) -> None:
        if value == self._mode:
            return
        current = self.detect if self._mode == "Detect" else self.train
        if current.is_busy():
            self.switch.set(self._mode)
            self.notice.configure(text="Finish the current recording or training before switching.")
            return
        self.notice.configure(text="")
        self._mode = value
        self._show(value)

    def _theme(self, value: str) -> None:
        apply_theme("dark" if value == "Dark" else "light")

    def _open_settings(self) -> None:
        def _saved(settings) -> None:
            self.detect.settings = settings
            self.detect._apply_recorder_settings()
            self.detect._refresh_hint()

        open_settings(self, _saved)

    def _tick(self) -> None:
        if self._closed:
            return
        if self._mode == "Detect":
            self.detect._tick()
        else:
            self.train._tick()
        self.after(40, self._tick)

    def _close(self) -> None:
        self._closed = True
        self.detect.shutdown()
        self.train.shutdown()
        sensor = self.sensor
        self.sensor = None
        if sensor is not None:
            sensor.close()
        self.destroy()


def run(mode: str = "detect") -> int:
    ctk.set_default_color_theme("blue")
    apply_theme(saved_theme(), persist=False)
    splash_status("Connecting to the Kinect.")
    sensor = None
    camera_error = ""
    try:
        sensor = KinectSensor()
    except Exception as exc:
        camera_error = str(exc)
    close_splash()
    KinectApp(sensor, camera_error, mode).mainloop()
    return 0
