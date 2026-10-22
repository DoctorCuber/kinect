"""Sleepwalking detector. Records infrared when a person walks into view."""

if __name__ == "__main__":
    from setup_runtime import ensure

    ensure()

import json
import shutil
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from kinect_view import KinectSensor, draw_joints, infrared_to_bgr
import person_check
from person_check import clear_clip_history, result_text, score_folder
from pose_check import classify_body
from report_window import ReportWindow
from settings_store import load as load_settings
import customtkinter as ctk

from ui_kit import PILL, SECONDARY, TEXT, AppButton, apply_list, bgr_image, body_font, fit_frame, open_settings, plate_box, segmented

GRACE_SECONDS = 120
PRE_SECONDS = 10
POST_SECONDS = 10
CLIP_FPS = 12
WALK_WINDOW_SECONDS = 1.2
WALK_DISTANCE_METERS = 0.4
MIN_STEP_METERS = 0.02
CLIPS_DIR = Path(__file__).resolve().parent / "clips"

SPINE_BASE = 0
SPINE_MID = 1
SPINE_SHOULDER = 20
LEG_JOINTS = (12, 13, 14, 16, 17, 18)


def person_present(bodies) -> bool:
    """A tracked body with enough joints is in view, even if they pause."""
    for body in bodies:
        seen = sum(1 for state in body.states if state > 0)
        if seen >= 5:
            return True
    return False


def _spine_xz(body):
    if not body.camera_xyz:
        return None
    for index in (SPINE_BASE, SPINE_MID, SPINE_SHOULDER):
        if body.states[index] != 2:
            continue
        x, _y, z = body.camera_xyz[index]
        if z > 0.4:
            return x, z
    return None


def _legs_visible(body) -> bool:
    return any(body.states[index] > 0 for index in LEG_JOINTS)


def _path_length(samples) -> float:
    total = 0.0
    previous = None
    for sample in samples:
        if previous is not None:
            step = ((sample[1] - previous[1]) ** 2 + (sample[2] - previous[2]) ** 2) ** 0.5
            if step >= MIN_STEP_METERS:
                total += step
        previous = sample
    return total


class SleepwalkRecorder:
    """Watch joint motion and write a clip with time before and after the walk."""

    def __init__(
        self,
        folder: Path = CLIPS_DIR,
        grace_seconds: float = GRACE_SECONDS,
        pre_seconds: float = PRE_SECONDS,
        post_seconds: float = POST_SECONDS,
        fps: int = CLIP_FPS,
    ) -> None:
        self.folder = Path(folder)
        self.grace_seconds = grace_seconds
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps
        self.walk_distance = WALK_DISTANCE_METERS
        self.min_body_span = 0.55
        self.lying_height = 0.45
        self.pose_counts = {"ok": 0, "cloud": 0, "lying": 0, "proportions": 0}
        self._last_pose_at = None
        self.phase = "idle"
        self.grace_until = 0.0
        self.buffer: deque[tuple[float, bytes]] = deque()
        self.walk_history: dict[int, deque[tuple[float, float, float]]] = {}
        self.writer = None
        self.clip_path: Path | None = None
        self.frames_written = 0
        self.lost_at: float | None = None
        self.last_sample: float | None = None
        self.last_raw: float | None = None
        self.raw_buffer: deque[bytes] = deque()
        self.clip_raw: list[bytes] = []
        self.saved: list[Path] = []
        self.note = "Press Start when the camera is aimed."

    @property
    def running(self) -> bool:
        return self.phase != "idle"

    def grace_remaining(self, now: float) -> float:
        return max(0.0, self.grace_until - now)

    def start(self, now: float) -> None:
        self.stop(now, discard_open_clip=True)
        self.phase = "grace"
        self.grace_until = now + self.grace_seconds
        self.buffer.clear()
        self.raw_buffer.clear()
        self.clip_raw = []
        self.walk_history.clear()
        self.pose_counts = {"ok": 0, "cloud": 0, "lying": 0, "proportions": 0}
        self._last_pose_at = None
        self.lost_at = None
        self.last_sample = None
        self.last_raw = None
        self.note = "Grace period. Leave the view before monitoring starts."

    def stop(self, now: float, discard_open_clip: bool = False) -> None:
        if self.writer is not None:
            self._finish_clip(discard_open_clip)
        self.phase = "idle"
        self.buffer.clear()
        self.raw_buffer.clear()
        self.clip_raw = []
        self.walk_history.clear()
        self.lost_at = None
        if discard_open_clip:
            return
        self.note = "Detector stopped."

    def update(self, now: float, bodies, frame, raw=None) -> None:
        if self.phase == "idle":
            return
        stored = self._remember(now, frame) if frame is not None else None
        if raw is not None:
            self._remember_raw(now, raw)
        walking = self._note_motion(now, bodies)
        present = self._plausible_present(bodies)
        self._sample_pose(now, bodies)

        if now < self.grace_until:
            self.phase = "grace"
            self.note = "Grace period. Movement is ignored until the countdown ends."
            return

        if self.phase != "recording":
            self.phase = "watching"
            self.note = "Monitoring for someone walking."
            if walking and self.buffer:
                self._begin_clip()
            return

        if stored is not None:
            self._write_jpeg(stored)
        if present:
            self.lost_at = None
            self.note = "Recording. A person is in view."
        else:
            if self.lost_at is None:
                self.lost_at = now
            remaining = self.post_seconds - (now - self.lost_at)
            self.note = f"Person left. Keeping {max(0, int(remaining))}s more."
            if now - self.lost_at >= self.post_seconds:
                self._finish_clip(False)
                self.phase = "watching"
                self.note = "Clip saved. Monitoring again."

    def _remember(self, now: float, frame) -> bytes | None:
        if self.last_sample is not None and now - self.last_sample < (1.0 / self.fps):
            return None
        self.last_sample = now
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        payload = encoded.tobytes()
        self.buffer.append((now, payload))
        cutoff = now - self.pre_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
        return payload

    def _note_motion(self, now: float, bodies) -> bool:
        seen = set()
        walking = False
        for body in bodies:
            if classify_body(body, self.min_body_span, self.lying_height) != "ok":
                continue
            if not _legs_visible(body):
                continue
            point = _spine_xz(body)
            if point is None:
                continue
            tracking_id = body.tracking_id or 1
            seen.add(tracking_id)
            history = self.walk_history.setdefault(tracking_id, deque())
            history.append((now, point[0], point[1]))
            while history and now - history[0][0] > WALK_WINDOW_SECONDS:
                history.popleft()
            span = history[-1][0] - history[0][0]
            if span >= 0.8 and _path_length(history) >= self.walk_distance:
                walking = True
        for tracking_id in list(self.walk_history):
            history = self.walk_history[tracking_id]
            if tracking_id not in seen and (not history or now - history[-1][0] > 2):
                del self.walk_history[tracking_id]
        return walking

    def _plausible_present(self, bodies) -> bool:
        for body in bodies:
            if sum(1 for state in body.states if state > 0) < 5:
                continue
            if classify_body(body, self.min_body_span, self.lying_height) == "ok":
                return True
        return False

    def _sample_pose(self, now: float, bodies) -> None:
        if self.phase != "recording":
            return
        if self._last_pose_at is not None and now - self._last_pose_at < 0.5:
            return
        self._last_pose_at = now
        kinds = []
        for body in bodies:
            if sum(1 for state in body.states if state > 0) < 5:
                continue
            kinds.append(classify_body(body, self.min_body_span, self.lying_height))
        if not kinds:
            return
        if "ok" in kinds:
            key = "ok"
        elif "lying" in kinds:
            key = "lying"
        elif "cloud" in kinds:
            key = "cloud"
        else:
            key = "proportions"
        self.pose_counts[key] += 1

    def _begin_clip(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.folder / f"sleepwalk_{stamp}.mp4"
        suffix = 2
        while path.exists():
            path = self.folder / f"sleepwalk_{stamp}_{suffix}.mp4"
            suffix += 1
        sample = cv2.imdecode(np.frombuffer(self.buffer[-1][1], np.uint8), cv2.IMREAD_COLOR)
        height, width = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            path = path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), self.fps, (width, height))
        if not writer.isOpened():
            self.note = "Could not create a video file."
            self.phase = "watching"
            return
        self.writer = writer
        self.clip_path = path
        self.frames_written = 0
        self.pose_counts = {"ok": 0, "cloud": 0, "lying": 0, "proportions": 0}
        self._last_pose_at = None
        self.clip_raw = list(self.raw_buffer)
        self.lost_at = None
        self.phase = "recording"
        for _stamp, payload in self.buffer:
            self._write_jpeg(payload)
        self.note = "Walking detected. Recording this pass, with buffer before and after."

    def _write_jpeg(self, payload: bytes) -> None:
        if self.writer is None:
            return
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.writer.write(frame)
        self.frames_written += 1

    def _finish_clip(self, discard: bool) -> None:
        writer = self.writer
        path = self.clip_path
        frames = self.frames_written
        self.writer = None
        self.clip_path = None
        self.frames_written = 0
        self.lost_at = None
        raw_frames = list(self.clip_raw)
        self.clip_raw = []
        if writer is not None:
            writer.release()
        if path is None:
            return
        pose_path = path.with_name(path.stem + "_pose.json")
        if discard or frames < 2:
            path.unlink(missing_ok=True)
            pose_path.unlink(missing_ok=True)
            shutil.rmtree(path.with_name(path.stem + "_frames"), ignore_errors=True)
            return
        self._write_raw_frames(path, raw_frames)
        pose_path.write_text(json.dumps(self.pose_counts), encoding="utf-8")
        self.saved.append(path)

    def _remember_raw(self, now: float, raw) -> None:
        if self.last_raw is not None and now - self.last_raw < 0.5:
            return
        self.last_raw = now
        ok, encoded = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        payload = encoded.tobytes()
        self.raw_buffer.append(payload)
        max_frames = int(self.pre_seconds * 2) + 2
        while len(self.raw_buffer) > max_frames:
            self.raw_buffer.popleft()
        if self.phase == "recording":
            self.clip_raw.append(payload)

    def _write_raw_frames(self, clip: Path, payloads: list[bytes]) -> None:
        if not payloads:
            return
        folder = clip.with_name(clip.stem + "_frames")
        folder.mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(payloads):
            (folder / f"{index:04d}.jpg").write_bytes(payload)


class DetectorApp(ctk.CTkFrame):
    def __init__(self, master, sensor: KinectSensor | None, camera_error: str = "") -> None:
        super().__init__(master, fg_color="transparent")
        self.sensor = sensor
        self.camera_error = camera_error
        self.settings = load_settings()
        self.recorder = SleepwalkRecorder(
            grace_seconds=self.settings.grace_seconds,
            pre_seconds=self.settings.pre_seconds,
            post_seconds=self.settings.post_seconds,
        )
        self._apply_recorder_settings()
        self._photo = None
        self._raw = None
        self._closed = False
        self.clip_results = {}
        self._saved_seen = 0
        self._load_saved_results()
        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True)

        ctk.CTkLabel(shell, text="Detect", text_color=TEXT, font=body_font(22, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(
            shell,
            text="The live picture can be infrared or color. Clips and the person check stay on infrared.",
            text_color=SECONDARY,
            font=body_font(13),
            anchor="w",
        ).pack(anchor="w", pady=(2, 8))

        self.view_mode = "infrared"
        self.view_switch = segmented(shell, ["Infrared + Joints", "Color + Joints"], self._on_view, height=28)
        self.view_switch.set("Infrared + Joints")
        self.view_switch.pack(anchor="w", pady=(0, 8))

        self.stage = ctk.CTkFrame(shell, fg_color="#0B0B0D", corner_radius=12)
        self.stage.pack(fill="both", expand=True)
        self.stage.pack_propagate(False)
        self.view = ctk.CTkLabel(self.stage, text="", fg_color="#0B0B0D")
        self.view.pack(expand=True, padx=8, pady=8)

        status_row = ctk.CTkFrame(shell, fg_color="transparent")
        status_row.pack(fill="x", pady=(12, 4))
        fill, ink = PILL["idle"]
        self.pill = ctk.CTkLabel(status_row, text="Idle", text_color=ink, fg_color=fill, corner_radius=8, font=body_font(12, "bold"), width=96, height=24)
        self.pill.pack(side="left")
        self.countdown = ctk.CTkLabel(status_row, text="", text_color=TEXT, font=body_font(13, "bold"))
        self.countdown.pack(side="left", padx=(12, 0))

        self.detail = ctk.CTkLabel(shell, text=self.recorder.note, text_color=SECONDARY, font=body_font(13), anchor="w")
        self.detail.pack(fill="x", pady=(4, 10))

        controls = ctk.CTkFrame(shell, fg_color="transparent")
        controls.pack(fill="x")
        self.start_button = AppButton(controls, text="Start Detector", command=self._start, kind="primary")
        self.start_button.pack(side="left")
        self.stop_button = AppButton(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.report_button = AppButton(controls, text="View Report", command=self._open_report, state="disabled")
        self.report_button.pack(side="left", padx=(8, 0))
        AppButton(controls, text="Open Clips", command=self._open_folder).pack(side="right")

        self.hint = ctk.CTkLabel(shell, text="", text_color=SECONDARY, font=body_font(12), wraplength=900, justify="left", anchor="w")
        self.hint.pack(anchor="w", pady=(10, 8))
        self._refresh_hint()

        ctk.CTkLabel(shell, text="Saved Clips", text_color=TEXT, font=body_font(15, "bold"), anchor="w").pack(anchor="w")
        self.clips = tk.Listbox(shell, height=5, highlightthickness=0, relief="flat", font=("Segoe UI", 12), activestyle="none")
        apply_list(self.clips)
        self.clips.pack(fill="x", pady=(4, 0))
        self._refresh_clips()
        if self.clip_results:
            self.report_button.configure(state="normal")

    def _on_view(self, value: str) -> None:
        self.view_mode = "color" if value.startswith("Color") else "infrared"

    def is_busy(self) -> bool:
        return self.recorder.running

    def _start(self) -> None:
        self.recorder.start(time.monotonic())
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def _stop(self) -> None:
        self.recorder.stop(time.monotonic())
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._check_clips()

    def _open_folder(self) -> None:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(CLIPS_DIR)

    def _tick(self) -> None:
        if self._closed:
            return
        if self.sensor is None:
            self.detail.configure(text=self.camera_error or "The Kinect is not open.")
            return
        now = time.monotonic()
        try:
            self.sensor.poll(self.view_mode == "color", True, True)
            raw = None if self.sensor.infrared is None else infrared_to_bgr(self.sensor.infrared, False)
            self._raw = raw
            if raw is not None:
                recorded = draw_joints(raw, self.sensor.bodies, "depth")
                self.recorder.update(now, self.sensor.bodies, recorded, raw)
                if self.view_mode == "color" and self.sensor.color is not None:
                    visible = draw_joints(self.sensor.color, self.sensor.bodies, "color")
                else:
                    visible = recorded
                box = plate_box(self.stage)
                if box is not None:
                    shown = fit_frame(visible, *box)
                    self._photo = bgr_image(shown)
                    self.view.configure(image=self._photo)
            else:
                self.recorder.update(now, self.sensor.bodies, None, None)
            self._paint_status(now)
        except Exception as exc:
            self.detail.configure(text=f"Frame error: {exc}")

    def _paint_status(self, now: float) -> None:
        phase = self.recorder.phase
        present = person_present(self.sensor.bodies)
        who = "Person in view" if present else "No person in view"
        if phase == "idle":
            self._set_pill("Idle", "idle")
            self.countdown.configure(text=who)
        elif phase == "grace":
            remaining = int(self.recorder.grace_remaining(now) + 0.999)
            minutes, seconds = divmod(remaining, 60)
            self._set_pill("Grace", "grace")
            self.countdown.configure(text=f"{minutes}:{seconds:02d}   {who}")
        elif phase == "recording":
            self._set_pill("Recording", "rec")
            self.countdown.configure(text=who)
        else:
            self._set_pill("Monitoring", "watch")
            self.countdown.configure(text=who)
        self.detail.configure(text=self.recorder.note)
        if len(self.recorder.saved) != self._saved_seen:
            self._saved_seen = len(self.recorder.saved)
            self._refresh_clips()

    def _set_pill(self, text: str, role: str) -> None:
        fill, ink = PILL[role]
        self.pill.configure(text=text, fg_color=fill, text_color=ink)

    def _refresh_clips(self) -> None:
        self.clips.delete(0, "end")
        paths = sorted(CLIPS_DIR.glob("sleepwalk_*.mp4"), reverse=True)
        paths += sorted(CLIPS_DIR.glob("sleepwalk_*.avi"), reverse=True)
        if not paths:
            self.clips.insert("end", "No clips yet.")
            return
        for path in paths:
            result = self.clip_results.get(path)
            self.clips.insert("end", result_text(result) if result else path.name)

    def _load_saved_results(self) -> None:
        import json

        if not CLIPS_DIR.exists():
            return
        for path in CLIPS_DIR.glob("sleepwalk_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.clip_results[Path(data["clip"])] = data
            except (OSError, KeyError, json.JSONDecodeError):
                continue

    def _check_clips(self, show_report: bool = True) -> None:
        self.detail.configure(text="Checking clips for a person...")
        self.update_idletasks()
        try:
            results = score_folder(CLIPS_DIR)
        except FileNotFoundError:
            self.recorder.note = (
                "Detector stopped. No person model yet. Label pictures in train_person_model.py, then train."
            )
            self.detail.configure(text=self.recorder.note)
            self._refresh_clips()
            return
        except Exception as exc:
            self.recorder.note = f"Detector stopped. Clip check failed: {exc}"
            self.detail.configure(text=self.recorder.note)
            self._refresh_clips()
            return
        self.clip_results = {Path(item["clip"]): item for item in results}
        people = sum(1 for item in results if item["label"] == "person")
        unfeasible = sum(1 for item in results if item["label"] == "unfeasible")
        if results:
            self.recorder.note = (
                f"Checked {len(results)} clips on this PC ({person_check.device_label}). "
                f"{people} person, {unfeasible} unfeasible."
            )
            self.report_button.style(state="normal")
            if show_report:
                self._open_report()
        else:
            self.recorder.note = "Detector stopped. No clips to check."
        self.detail.configure(text=self.recorder.note)
        self._refresh_clips()

    def _apply_recorder_settings(self) -> None:
        self.recorder.grace_seconds = self.settings.grace_seconds
        self.recorder.pre_seconds = self.settings.pre_seconds
        self.recorder.post_seconds = self.settings.post_seconds
        self.recorder.walk_distance = self.settings.walk_distance_meters
        self.recorder.min_body_span = self.settings.min_body_span_meters
        self.recorder.lying_height = self.settings.lying_height_meters

    def _refresh_hint(self) -> None:
        settings = self.settings
        self.hint.configure(
            text=(
                f"After Start there is a {int(settings.grace_seconds)} second grace period. "
                f"A walk of {settings.walk_distance_meters:.1f} m is saved with {int(settings.pre_seconds)}s before it "
                f"and {int(settings.post_seconds)}s after the person leaves. "
                "Person means a standing human. Lying down is Unfeasible, and so is a short joint cloud or impossible limb lengths. "
                "Stop scores the clips and opens the report."
            )
        )

    def _open_settings(self) -> None:
        def _saved(settings) -> None:
            self.settings = settings
            self._apply_recorder_settings()
            self._refresh_hint()

        open_settings(self, _saved)

    def _open_report(self) -> None:
        results = list(self.clip_results.values())
        if results:
            ReportWindow(self, results, on_cleared=self._history_cleared)

    def _history_cleared(self) -> None:
        keep = [self.recorder.clip_path] if self.recorder.clip_path is not None else []
        clear_clip_history(keep=keep)
        self.clip_results = {}
        self.recorder.saved.clear()
        self._saved_seen = 0
        self.report_button.configure(state="disabled")
        self._refresh_clips()
        self.detail.configure(text="Report history cleared. Training pictures and the person model were kept.")

    def shutdown(self) -> None:
        self._closed = True
        was_running = self.recorder.running
        self.recorder.stop(time.monotonic())
        if was_running:
            self._check_clips(show_report=False)


def main() -> int:
    try:
        from kinect_app import run

        return run("detect")
    except Exception as exc:
        from setup_runtime import close_splash

        close_splash()
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Kinect", f"The window could not open.\n\n{exc}")
        root.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
