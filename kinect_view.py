"""Kinect v2 viewer: color, infrared, and joint tracking in one window."""

if __name__ == "__main__":
    from setup_runtime import ensure

    ensure()

import ctypes
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import cv2
import numpy as np
from comtypes import COMError

E_PENDING = 0x8000000A
PANEL_HEIGHT = 480
JOINT_COUNT = 25
_prepared = False

# Kinect v2 joint indexes. Bones follow the standard skeleton.
BONES = (
    (0, 1),
    (1, 20),
    (20, 2),
    (2, 3),
    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    (6, 22),
    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (10, 24),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
)
BODY_COLORS = (
    (80, 220, 80),
    (40, 180, 255),
    (255, 160, 40),
    (220, 80, 220),
    (80, 220, 220),
    (60, 80, 255),
)


def _prepare_pykinect2() -> None:
    """Make the pykinect2 0.1.0 binding importable on 64-bit Python."""
    global _prepared
    if _prepared:
        return
    _prepared = True

    import pathlib

    import comtypes
    import comtypes._tlib_version_checker as checker

    checker._check_version = lambda *args, **kwargs: None
    comtypes._check_version = checker._check_version

    path = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "pykinect2" / "PyKinectV2.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    old = "assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)"
    new = "assert sizeof(tagSTATSTG) in (72, 80), sizeof(tagSTATSTG)"
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _import_kinect():
    _prepare_pykinect2()
    from pykinect2 import PyKinectV2

    return PyKinectV2


def _pending(exc: COMError) -> bool:
    return (exc.hresult & 0xFFFFFFFF) == E_PENDING


def _acquire(reader):
    try:
        return reader.AcquireLatestFrame()
    except COMError as exc:
        if _pending(exc):
            return None
        raise


@dataclass
class BodyPose:
    states: list[int]
    color_xy: list[tuple[float, float]]
    depth_xy: list[tuple[float, float]]
    camera_xyz: list[tuple[float, float, float]] | None = None
    tracking_id: int = 0


class KinectSensor:
    """Color, infrared, and body readers on one Kinect v2 sensor."""

    def __init__(self) -> None:
        kinect = _import_kinect()
        sensor = ctypes.POINTER(kinect.IKinectSensor)()
        hr = kinect.DefaultKinectSensor._GetDefaultKinectSensorProto(ctypes.byref(sensor))
        if hr != 0 or not sensor:
            raise RuntimeError(f"No Kinect sensor was found (HRESULT {hr:#010x}).")

        sensor.Open()
        deadline = time.perf_counter() + 8
        while not bool(sensor.IsAvailable) and time.perf_counter() < deadline:
            time.sleep(0.1)

        self._kinect = kinect
        self.sensor = sensor
        self.mapper = sensor.CoordinateMapper

        color_source = sensor.ColorFrameSource
        color_desc = color_source.FrameDescription
        self.color_width = int(color_desc.Width)
        self.color_height = int(color_desc.Height)
        self.color_reader = color_source.OpenReader()
        self._color_buffer = (ctypes.c_ubyte * (self.color_width * self.color_height * 4))()

        infrared_source = sensor.InfraredFrameSource
        infrared_desc = infrared_source.FrameDescription
        self.infrared_width = int(infrared_desc.Width)
        self.infrared_height = int(infrared_desc.Height)
        self.infrared_reader = infrared_source.OpenReader()
        self._infrared_buffer = (ctypes.c_ushort * (self.infrared_width * self.infrared_height))()

        body_source = sensor.BodyFrameSource
        self.body_count = int(body_source.BodyCount)
        self.body_reader = body_source.OpenReader()

        self.color = None
        self.infrared = None
        self.bodies: list[BodyPose] = []
        self._bind_mapper()

    def _bind_mapper(self) -> None:
        # comtypes passes these 12-byte camera points with the wrong calling
        # convention on 64-bit Windows, and the mapper answers with off-screen
        # coordinates. Call the COM methods directly instead.
        kinect = self._kinect
        header = ctypes.cast(self.mapper, ctypes.POINTER(ctypes.c_void_p))
        table = ctypes.cast(header[0], ctypes.POINTER(ctypes.c_void_p))
        self._mapper_this = ctypes.cast(self.mapper, ctypes.c_void_p).value
        color_type = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(kinect._CameraSpacePoint),
            ctypes.c_uint,
            ctypes.POINTER(kinect._ColorSpacePoint),
        )
        depth_type = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(kinect._CameraSpacePoint),
            ctypes.c_uint,
            ctypes.POINTER(kinect._DepthSpacePoint),
        )
        self._map_color_points = color_type(table[11])
        self._map_depth_points = depth_type(table[10])

    def poll(self, want_color: bool, want_infrared: bool, want_joints: bool) -> None:
        if want_color:
            frame = self._read_color()
            if frame is not None:
                self.color = frame
        # An infrared frame has to arrive before the coordinate mapper will
        # place joints on the color or infrared image.
        if want_infrared or want_joints:
            frame = self._read_infrared()
            if frame is not None:
                self.infrared = frame
        if want_joints:
            bodies = self._read_bodies()
            if bodies is not None:
                self.bodies = bodies

    def close(self) -> None:
        self.color_reader = None
        self.infrared_reader = None
        self.body_reader = None
        self.mapper = None
        sensor = self.sensor
        self.sensor = None
        if sensor is not None:
            sensor.Close()

    def _read_color(self):
        frame = _acquire(self.color_reader)
        if not frame:
            return None
        try:
            frame.CopyConvertedFrameDataToArray(
                self.color_width * self.color_height * 4,
                ctypes.cast(self._color_buffer, ctypes.POINTER(ctypes.c_ubyte)),
                self._kinect.ColorImageFormat_Bgra,
            )
        finally:
            del frame
        image = np.ctypeslib.as_array(self._color_buffer)
        image = image.reshape(self.color_height, self.color_width, 4)
        return np.ascontiguousarray(image[:, :, :3])

    def _read_infrared(self):
        frame = _acquire(self.infrared_reader)
        if not frame:
            return None
        try:
            frame.CopyFrameDataToArray(
                self.infrared_width * self.infrared_height,
                ctypes.cast(self._infrared_buffer, ctypes.POINTER(ctypes.c_ushort)),
            )
        finally:
            del frame
        image = np.ctypeslib.as_array(self._infrared_buffer)
        return np.array(image.reshape(self.infrared_height, self.infrared_width), copy=True)

    def _read_bodies(self):
        frame = _acquire(self.body_reader)
        if not frame:
            return None
        kinect = self._kinect
        capacity = self.body_count
        storage = (ctypes.POINTER(kinect.IBody) * capacity)()
        body_data = ctypes.cast(storage, ctypes.POINTER(ctypes.POINTER(kinect.IBody)))
        try:
            frame.GetAndRefreshBodyData(capacity, body_data)
            people = []
            for index in range(capacity):
                body = body_data[index]
                if not body or not bool(body.IsTracked):
                    continue
                joint_values = (kinect._Joint * JOINT_COUNT)()
                joints = ctypes.cast(joint_values, ctypes.POINTER(kinect._Joint))
                body.GetJoints(JOINT_COUNT, joints)
                try:
                    tracking_id = int(body.TrackingId)
                except Exception:
                    tracking_id = 0
                states = []
                positions = []
                for joint_index in range(JOINT_COUNT):
                    joint = joints[joint_index]
                    states.append(int(joint.TrackingState))
                    position = joint.Position
                    positions.append((float(position.x), float(position.y), float(position.z)))
                color_xy, depth_xy = self._map_positions(positions)
                people.append(BodyPose(states, color_xy, depth_xy, positions, tracking_id))
            return people
        finally:
            del frame

    def _map_positions(self, positions):
        kinect = self._kinect
        count = len(positions)
        cameras = (kinect._CameraSpacePoint * count)()
        colors = (kinect._ColorSpacePoint * count)()
        depths = (kinect._DepthSpacePoint * count)()
        for index, (x, y, z) in enumerate(positions):
            cameras[index].x = x
            cameras[index].y = y
            cameras[index].z = z
        camera_ptr = ctypes.cast(cameras, ctypes.POINTER(kinect._CameraSpacePoint))
        self._map_color_points(
            self._mapper_this,
            count,
            camera_ptr,
            count,
            ctypes.cast(colors, ctypes.POINTER(kinect._ColorSpacePoint)),
        )
        self._map_depth_points(
            self._mapper_this,
            count,
            camera_ptr,
            count,
            ctypes.cast(depths, ctypes.POINTER(kinect._DepthSpacePoint)),
        )
        color_xy = [(float(colors[index].x), float(colors[index].y)) for index in range(count)]
        depth_xy = [(float(depths[index].x), float(depths[index].y)) for index in range(count)]
        return color_xy, depth_xy


def infrared_to_bgr(frame: np.ndarray, colorize: bool) -> np.ndarray:
    """Turn the 16-bit infrared frame into an 8-bit picture."""
    scaled = np.clip(frame.astype(np.float32) * (255.0 / 8000.0), 0, 255).astype(np.uint8)
    if colorize:
        return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def _point_ok(x: float, y: float) -> bool:
    return np.isfinite(x) and np.isfinite(y) and abs(x) < 10000 and abs(y) < 10000


def draw_joints(image: np.ndarray, bodies: list[BodyPose], space: str) -> np.ndarray:
    canvas = image.copy()
    tracked = 2
    for body_index, body in enumerate(bodies):
        points = body.color_xy if space == "color" else body.depth_xy
        color = BODY_COLORS[body_index % len(BODY_COLORS)]
        for start, end in BONES:
            if body.states[start] == 0 or body.states[end] == 0:
                continue
            x1, y1 = points[start]
            x2, y2 = points[end]
            if not (_point_ok(x1, y1) and _point_ok(x2, y2)):
                continue
            thickness = 2 if body.states[start] == tracked and body.states[end] == tracked else 1
            cv2.line(
                canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                thickness,
                cv2.LINE_AA,
            )
        for index, (x, y) in enumerate(points):
            state = body.states[index]
            if state == 0 or not _point_ok(x, y):
                continue
            center = (int(round(x)), int(round(y)))
            if state == tracked:
                cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, center, 5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, center, 4, color, 1, cv2.LINE_AA)
    return canvas


def _scale_to_height(image: np.ndarray, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    if source_height == height:
        return image
    width = max(1, int(round(source_width * (height / source_height))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, 0.65, 1)
    cv2.rectangle(image, (8, 8), (20 + width, 18 + height + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (14, 14 + height), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def _placeholder(width: int, height: int, text: str) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(image, text, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)
    return image


def compose(sensor: KinectSensor, show_color: bool, show_infrared: bool, show_joints: bool, colorize: bool):
    """Build the picture for the current checkbox combination."""
    panels = []
    if show_color:
        if sensor.color is None:
            panel = _placeholder(sensor.color_width, sensor.color_height, "Waiting for color")
        elif show_joints:
            panel = draw_joints(sensor.color, sensor.bodies, "color")
        else:
            panel = sensor.color.copy()
        _label(panel, "Color + joints" if show_joints else "Color")
        panels.append(panel)

    show_infrared_panel = show_infrared or (show_joints and not show_color)
    if show_infrared_panel:
        if sensor.infrared is None:
            panel = _placeholder(sensor.infrared_width, sensor.infrared_height, "Waiting for infrared")
        else:
            panel = infrared_to_bgr(sensor.infrared, colorize)
        if show_joints:
            panel = draw_joints(panel, sensor.bodies, "depth")
        if show_joints:
            caption = "Infrared + joints"
        else:
            caption = "Infrared"
        _label(panel, caption)
        panels.append(panel)

    if not panels:
        return _placeholder(640, PANEL_HEIGHT, "Turn on Color, Infrared, or Joints")

    fitted = [_scale_to_height(panel, PANEL_HEIGHT) for panel in panels]
    if len(fitted) == 1:
        return fitted[0]
    gap = np.zeros((PANEL_HEIGHT, 12, 3), dtype=np.uint8)
    output = fitted[0]
    for panel in fitted[1:]:
        output = np.hstack((output, gap, panel))
    return output


def bgr_to_photo(image: np.ndarray) -> tk.PhotoImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


class Viewer(tk.Tk):
    def __init__(self, sensor: KinectSensor) -> None:
        super().__init__()
        self.sensor = sensor
        self.title("Kinect")
        self.configure(bg="#1e1e1e")
        self.minsize(860, 560)
        self._photo = None
        self._closed = False

        self.show_color = tk.BooleanVar(value=True)
        self.show_infrared = tk.BooleanVar(value=False)
        self.show_joints = tk.BooleanVar(value=False)
        self.colorize = tk.BooleanVar(value=False)

        bar = ttk.Frame(self, padding=(10, 8, 10, 4))
        bar.pack(fill="x")
        presets = (
            ("Color", True, False, False),
            ("Infrared", False, True, False),
            ("Infrared + joints", False, True, True),
            ("Color + joints", True, False, True),
            ("All three", True, True, True),
        )
        for label, color, infrared, joints in presets:
            ttk.Button(
                bar,
                text=label,
                command=lambda c=color, i=infrared, j=joints: self._preset(c, i, j),
            ).pack(side="left", padx=(0, 6))

        checks = ttk.Frame(self, padding=(10, 0, 10, 6))
        checks.pack(fill="x")
        ttk.Checkbutton(checks, text="Color", variable=self.show_color).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(checks, text="Infrared", variable=self.show_infrared).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(checks, text="Joints", variable=self.show_joints).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(checks, text="Colorize infrared", variable=self.colorize).pack(side="left")

        self.status = ttk.Label(self, padding=(10, 0, 10, 6), anchor="w")
        self.status.pack(fill="x")

        stage = tk.Frame(self, bg="black")
        stage.pack(fill="both", expand=True)
        self.view = tk.Label(stage, bg="black", bd=0)
        self.view.pack(expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(20, self._tick)

    def _preset(self, color: bool, infrared: bool, joints: bool) -> None:
        self.show_color.set(color)
        self.show_infrared.set(infrared)
        self.show_joints.set(joints)

    def _tick(self) -> None:
        if self._closed:
            return
        show_color = bool(self.show_color.get())
        show_infrared = bool(self.show_infrared.get())
        show_joints = bool(self.show_joints.get())
        try:
            self.sensor.poll(show_color, show_infrared, show_joints)
            picture = compose(self.sensor, show_color, show_infrared, show_joints, bool(self.colorize.get()))
            self._photo = bgr_to_photo(picture)
            self.view.configure(image=self._photo)
            self.status.configure(text=self._status(show_joints))
        except Exception as exc:
            self.status.configure(text=f"Frame error: {exc}")
        self.after(20, self._tick)

    def _status(self, show_joints: bool) -> str:
        parts = [
            f"Color {self.sensor.color_width}x{self.sensor.color_height}",
            f"Infrared {self.sensor.infrared_width}x{self.sensor.infrared_height}",
        ]
        if show_joints:
            count = len(self.sensor.bodies)
            noun = "body" if count == 1 else "bodies"
            parts.append(f"{count} {noun}")
        return "  ·  ".join(parts)

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.sensor.close()
        finally:
            self.destroy()


def main() -> int:
    try:
        _import_kinect()
    except ImportError:
        print(
            "pykinect2 is not installed in this environment.\n"
            "From the project folder, run:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(
            "The Kinect SDK runtime (Kinect20.dll) is not installed.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    from setup_runtime import close_splash

    try:
        sensor = KinectSensor()
    except Exception as exc:
        close_splash()
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Kinect",
            "Could not open the Kinect.\n"
            "The sensor needs the Kinect for Windows SDK 2.0 driver and a USB 3.0 port.\n"
            f"Details: {exc}",
        )
        root.destroy()
        return 1

    close_splash()
    app = Viewer(sensor)
    app.lift()
    app.attributes("-topmost", True)
    app.after(400, lambda: app.attributes("-topmost", False))
    app.focus_force()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
