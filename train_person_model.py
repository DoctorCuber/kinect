"""Train the infrared classifier from standing, unfeasible, and empty pictures."""

if __name__ == "__main__":
    from setup_runtime import ensure

    ensure()

import random
import threading
import time
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from compute_device import choose_compute
from kinect_view import KinectSensor, draw_joints, infrared_to_bgr
from person_check import DATASET_DIR, MODEL_PATH
from settings_store import load as load_settings
from ui_kit import (
    ACCENT,
    ACCENT_HOVER,
    ACCENT_TEXT,
    CHIP,
    GLASS_EDGE,
    PILL,
    SECONDARY,
    SURFACE,
    TEXT,
    WELL,
    AppButton,
    bgr_image,
    body_font,
    fit_frame,
    glass_frame,
    plate_box,
    segmented,
    wash_frame,
)

VIDEO_FPS = 12
VIDEO_DIR = DATASET_DIR / "videos"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_PER_CLASS = 15
CLASSES = ("no_person", "person", "unfeasible")
LABELS = (("Person", "person"), ("Empty Room", "no_person"), ("Unfeasible", "unfeasible"))
LABEL_TITLE = {kind: title for title, kind in LABELS}
LABEL_KIND = {title: kind for title, kind in LABELS}
LABEL_NAME = {"person": "standing person", "no_person": "empty room", "unfeasible": "unfeasible"}
VIDEO_PREFIX = {"person": "person", "no_person": "empty", "unfeasible": "unfeasible"}


class InfraredDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)), label


def collect_samples() -> dict[str, list[Path]]:
    found = {}
    for name in CLASSES:
        folder = DATASET_DIR / name
        files = []
        if folder.exists():
            files = [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        found[name] = sorted(files)
    return found


def split_samples(found: dict[str, list[Path]], seed: int = 7):
    random.seed(seed)
    train, test = [], []
    use_validation = all(len(paths) >= 20 for paths in found.values())
    for label, name in enumerate(CLASSES):
        paths = found[name][:]
        random.shuffle(paths)
        if use_validation:
            cut = max(1, int(len(paths) * 0.2))
            test.extend((path, label) for path in paths[:cut])
            train.extend((path, label) for path in paths[cut:])
        else:
            train.extend((path, label) for path in paths)
    return train, test


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    last_linear = max(index for index, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear))
    in_features = model.classifier[last_linear].in_features
    model.classifier[last_linear] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def train(log=print) -> int:
    found = collect_samples()
    counts = {name: len(paths) for name, paths in found.items()}
    log(
        f"Dataset: person {counts['person']}, unfeasible {counts['unfeasible']}, empty {counts['no_person']}"
    )
    short = [LABEL_NAME[name] for name, count in counts.items() if count < MIN_PER_CLASS]
    if short:
        log(
            f"Need at least {MIN_PER_CLASS} pictures in each folder. "
            f"Still short: {', '.join(short)}."
        )
        log("Choose Person, Unfeasible, or Empty Room, press Photo or Video, and wait out the countdown.")
        return 1

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    train_samples, test_samples = split_samples(found)
    train_loader = DataLoader(
        InfraredDataset(train_samples, transform),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    test_loader = None
    if test_samples:
        test_loader = DataLoader(
            InfraredDataset(test_samples, transform),
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )

    kind, label, train_device = choose_compute()
    device = torch.device(train_device)
    if kind == "npu":
        log(f"No NVIDIA GPU. Training on CPU. Clip checking will use {label}.")
    else:
        log(f"Training on {label}")
    model = build_model(len(CLASSES)).to(device)
    counts_list = [counts[name] for name in CLASSES]
    class_weights = torch.tensor(
        [sum(counts_list) / count for count in counts_list],
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    for epoch in range(1, 7):
        loss, accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        log(f"Head epoch {epoch}: loss {loss:.3f}  train accuracy {accuracy:.1%}")

    for parameter in model.features.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1, 7):
        loss, accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        message = f"full epoch {epoch}: loss {loss:.3f}  train accuracy {accuracy:.1%}"
        if test_loader is not None:
            test_loss, test_accuracy = run_epoch(model, test_loader, criterion, None, device)
            message += f"  validation accuracy {test_accuracy:.1%}"
        log(message)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": list(CLASSES)},
        MODEL_PATH,
    )
    log(f"Saved {MODEL_PATH}")
    if test_loader is None:
        log("Validation was skipped because each folder has fewer than 20 pictures. Add more, then train again.")
    log("Press Stop in the sleepwalk detector to score the saved clips.")
    return 0


def _load_trained(model: nn.Module) -> bool:
    """Start from the supervised checkpoint when its classes match this trainer."""
    if not MODEL_PATH.exists():
        return False
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if list(checkpoint.get("classes", [])) != list(CLASSES):
        return False
    model.load_state_dict(checkpoint["state_dict"])
    return True


def reinforce_epoch(model, loader, optimizer, device, baseline: float, class_weights) -> tuple[float, float, float]:
    """One pass of REINFORCE.

    The action is a class sampled from the model. The reward is the class weight
    when that guess matches the folder the photo came from, and zero otherwise.
    The baseline is a moving average of reward so the update follows advantage.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_reward = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        distribution = torch.distributions.Categorical(logits=logits)
        if training:
            actions = distribution.sample()
            reward = (actions == labels).float() * class_weights[labels]
            advantage = reward - baseline
            loss = -(distribution.log_prob(actions) * advantage.detach()).mean()
            loss = loss - 0.01 * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            baseline = 0.95 * baseline + 0.05 * float(reward.mean().item())
            total_loss += float(loss.item()) * labels.size(0)
        else:
            actions = logits.argmax(dim=1)
            reward = (actions == labels).float()
        total_reward += float(reward.sum().item())
        total += int(labels.size(0))
    return total_loss / max(total, 1), total_reward / max(total, 1), baseline


def reinforce_train(log=print) -> int:
    """Fine-tune the room classifier with a reward, using the captured pictures."""
    found = collect_samples()
    counts = {name: len(paths) for name, paths in found.items()}
    log(
        f"Dataset: person {counts['person']}, unfeasible {counts['unfeasible']}, empty {counts['no_person']}"
    )
    short = [LABEL_NAME[name] for name, count in counts.items() if count < MIN_PER_CLASS]
    if short:
        log(
            f"Need at least {MIN_PER_CLASS} pictures in each folder. "
            f"Still short: {', '.join(short)}."
        )
        return 1

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    train_samples, test_samples = split_samples(found)
    train_loader = DataLoader(
        InfraredDataset(train_samples, transform),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    test_loader = None
    if test_samples:
        test_loader = DataLoader(
            InfraredDataset(test_samples, transform),
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )

    kind, label, train_device = choose_compute()
    device = torch.device(train_device)
    if kind == "npu":
        log(f"No NVIDIA GPU. Reinforcement training on CPU. Clip checking will use {label}.")
    else:
        log(f"Reinforcement training on {label}")
    model = build_model(len(CLASSES)).to(device)
    if _load_trained(model):
        log(f"Starting from the supervised checkpoint {MODEL_PATH.name}.")
    else:
        log("No matching supervised checkpoint. Starting from ImageNet weights.")
    counts_list = [counts[name] for name in CLASSES]
    class_weights = torch.tensor(
        [sum(counts_list) / count for count in counts_list],
        dtype=torch.float,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    baseline = 0.0
    for epoch in range(1, 9):
        loss, reward, baseline = reinforce_epoch(
            model, train_loader, optimizer, device, baseline, class_weights
        )
        message = f"reinforce epoch {epoch}: loss {loss:.3f}  mean reward {reward:.3f}  baseline {baseline:.3f}"
        if test_loader is not None:
            _test_loss, accuracy, _baseline = reinforce_epoch(
                model, test_loader, None, device, baseline, class_weights
            )
            message += f"  validation accuracy {accuracy:.1%}"
        log(message)

    destination = MODEL_PATH.with_name("person_classifier_rl.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": list(CLASSES), "method": "reinforce"},
        destination,
    )
    log(f"Saved {destination}")
    log("The supervised model file was left unchanged.")
    return 0


def _stamp(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    origin = (16, 72)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 2, cv2.LINE_AA)
    return canvas


class TrainerApp(ctk.CTkFrame):
    def __init__(self, master, sensor: KinectSensor | None = None, camera_error: str = "") -> None:
        super().__init__(master, fg_color="transparent")
        self.sensor = sensor
        self._training = False
        self._closed = False
        self._photo = None
        self._raw = None
        self.label_kind = "person"
        self.armed_for = load_settings().shutter_seconds
        self.phase = "idle"
        self.arm_at = 0.0
        self.capture_label = "person"
        self.writer = None
        self.video_path = None
        self.frames_written = 0
        self.stills_written = 0
        self.last_video_frame = None
        self.last_still = None

        shell = wash_frame(self)
        shell.pack(fill="both", expand=True)
        ctk.CTkLabel(shell, text="Train", text_color=TEXT, font=body_font(22, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(
            shell,
            text="Person is someone standing. Unfeasible is someone lying down, or otherwise not standing. "
            "Empty Room is no one. The saved pictures are plain infrared, with no skeleton drawn on them.",
            text_color=SECONDARY,
            font=body_font(13),
            wraplength=920,
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(2, 10))

        self.stage = glass_frame(shell, corner_radius=28)
        self.stage.pack(fill="both", expand=True)
        self.stage.pack_propagate(False)
        self.view = ctk.CTkLabel(self.stage, text="", fg_color=WELL, corner_radius=16)
        self.view.pack(expand=True, padx=12, pady=12)

        self.status = ctk.CTkLabel(shell, text="", text_color=TEXT, font=body_font(13), anchor="w")
        self.status.pack(fill="x", pady=(8, 8))

        self.label_switch = segmented(shell, [title for title, _kind in LABELS], self._on_label, height=28)
        self.label_switch.set("Person")
        self.label_switch.pack(anchor="w")

        row = wash_frame(shell)
        row.pack(fill="x", pady=(8, 0))
        self.photo_button = AppButton(row, text="Photo", command=self._photo_shutter)
        self.photo_button.pack(side="left")
        self.video_button = AppButton(row, text="Video", command=self._video_shutter)
        self.video_button.pack(side="left", padx=(8, 0))
        self.train_button = AppButton(row, text="Train", command=self._start, kind="primary")
        self.train_button.pack(side="left", padx=(8, 0))
        AppButton(row, text="Refresh Counts", command=self._refresh).pack(side="left", padx=(8, 0))

        self.counts = ctk.CTkLabel(shell, text="", text_color=TEXT, font=body_font(13), anchor="w")
        self.counts.pack(anchor="w", pady=(10, 0))
        self.model_label = ctk.CTkLabel(shell, text="", text_color=SECONDARY, font=body_font(12), anchor="w")
        self.model_label.pack(anchor="w", pady=(2, 8))

        self.log = ctk.CTkTextbox(
            shell,
            height=160,
            fg_color=SURFACE,
            text_color=TEXT,
            font=("Segoe UI", 12),
            wrap="word",
            corner_radius=16,
            border_width=1,
            border_color=GLASS_EDGE,
            scrollbar_button_color=ACCENT,
            scrollbar_button_hover_color=ACCENT_HOVER,
        )
        self.log.pack(fill="x", pady=(0, 0))
        self.log.configure(state="disabled")
        self._refresh()
        self._style_capture_buttons()
        if camera_error:
            self.status.configure(text="Kinect is not open. Training still works with pictures you already have.")
            self._append(f"Could not open the Kinect: {camera_error}")
        elif sensor is None:
            self.status.configure(text="No camera. Training still works with pictures you already have.")
        else:
            self.status.configure(text="Live infrared. Person means standing. Unfeasible means lying down.")

    def is_busy(self) -> bool:
        return self._training or self.phase != "idle"

    def _on_label(self, value: str) -> None:
        self._set_label(LABEL_KIND[value])

    def _set_label(self, kind: str) -> None:
        if self.phase != "idle" or self._training:
            self.label_switch.set(LABEL_TITLE[self.label_kind])
            return
        self.label_kind = kind
        self.status.configure(text=f"Next photo or video will be labeled {self._label_name(kind)}.")

    def _style_capture_buttons(self) -> None:
        quiet = {"fg_color": CHIP, "text_color": TEXT, "state": "normal"}
        primary = {"fg_color": ACCENT, "text_color": ACCENT_TEXT, "state": "normal"}
        grace_fill, grace_ink = PILL["grace"]
        rec_fill, rec_ink = PILL["rec"]
        self.photo_button.style(text="Photo", **quiet)
        self.video_button.style(text="Video", **quiet)
        self.train_button.style(text="Train", **primary)
        if self._training:
            self.photo_button.style(state="disabled")
            self.video_button.style(state="disabled")
            self.train_button.style(text="Training...", state="disabled", fg_color=CHIP, text_color=SECONDARY)
            return
        if self.sensor is None:
            self.photo_button.style(state="disabled")
            self.video_button.style(state="disabled")
        if self.phase == "photo_wait":
            self.photo_button.style(text="Cancel Photo", fg_color=grace_fill, text_color=grace_ink)
            self.video_button.style(state="disabled")
            self.train_button.style(state="disabled", fg_color=CHIP, text_color=SECONDARY)
        elif self.phase == "video_wait":
            self.video_button.style(text="Cancel Video", fg_color=grace_fill, text_color=grace_ink)
            self.photo_button.style(state="disabled")
            self.train_button.style(state="disabled", fg_color=CHIP, text_color=SECONDARY)
        elif self.phase == "video_record":
            self.video_button.style(text="Stop Video", fg_color=rec_fill, text_color=rec_ink)
            self.photo_button.style(state="disabled")
            self.train_button.style(state="disabled", fg_color=CHIP, text_color=SECONDARY)

    def _label_name(self, kind: str) -> str:
        return LABEL_NAME[kind]

    def _photo_shutter(self) -> None:
        if self._training or self.sensor is None:
            return
        if self.phase == "photo_wait":
            self._idle_capture("Photo cancelled.")
            return
        if self.phase != "idle":
            return
        self._arm("photo_wait")

    def _video_shutter(self) -> None:
        if self._training or (self.sensor is None and self.phase == "idle"):
            return
        if self.phase == "video_wait":
            self._idle_capture("Video cancelled.")
            return
        if self.phase == "video_record":
            self._stop_video()
            return
        if self.phase != "idle":
            return
        self._arm("video_wait")

    def _arm(self, phase: str) -> None:
        self.phase = phase
        self.arm_at = time.monotonic()
        self.armed_for = max(0.0, load_settings().shutter_seconds)
        self.capture_label = self.label_kind
        kind = "photo" if phase == "photo_wait" else "video"
        self.status.configure(
            text=f"Get in place. {kind.capitalize()} in {int(self.armed_for)} seconds. Labeled {self._label_name(self.capture_label)}."
        )
        self._style_capture_buttons()

    def _idle_capture(self, message: str = "") -> None:
        self.phase = "idle"
        self.writer = None
        self.video_path = None
        if message:
            self.status.configure(text=message)
            self._append(message)
        self._style_capture_buttons()

    def _tick(self) -> None:
        if self._closed:
            return
        now = time.monotonic()
        raw = None
        if self.sensor is not None:
            try:
                self.sensor.poll(False, True, True)
                if self.sensor.infrared is not None:
                    raw = infrared_to_bgr(self.sensor.infrared, False)
            except Exception as exc:
                self.status.configure(text=f"Frame error: {exc}")
        self._raw = raw
        self._handle_capture(now, raw)
        self._show(raw, now)

    def _handle_capture(self, now: float, raw) -> None:
        if self.phase in {"photo_wait", "video_wait"}:
            if now - self.arm_at < self.armed_for or raw is None:
                return
            if self.phase == "photo_wait":
                path = self._save_still(raw, self.capture_label)
                self._append(f"Saved {self._label_name(self.capture_label)} photo {path.name}")
                self._refresh()
                self.phase = "idle"
                self.status.configure(text=f"Saved one {self._label_name(self.capture_label)} photo.")
                self._style_capture_buttons()
            else:
                self._open_writer(raw)
            return
        if self.phase != "video_record" or raw is None or self.writer is None:
            return
        if self.last_video_frame is None or now - self.last_video_frame >= (1.0 / VIDEO_FPS):
            self.writer.write(raw)
            self.frames_written += 1
            self.last_video_frame = now
        if self.last_still is None or now - self.last_still >= 1.0:
            self._save_still(raw, self.capture_label)
            self.stills_written += 1
            self.last_still = now
            self._refresh()

    def _show(self, raw, now: float) -> None:
        if raw is None:
            return
        visible = draw_joints(raw, self.sensor.bodies, "depth") if self.sensor is not None else raw
        if self.phase in {"photo_wait", "video_wait"}:
            remaining = int(self.armed_for) - int(now - self.arm_at)
            visible = _stamp(visible, str(max(remaining, 1)), (227, 113, 0))
            self.status.configure(
                text=f"Get in place. {max(remaining, 1)} seconds. Labeled {self._label_name(self.capture_label)}."
            )
        elif self.phase == "video_record":
            visible = _stamp(visible, "REC", (227, 113, 0))
            self.status.configure(
                text=f"Recording {self._label_name(self.capture_label)}. {self.stills_written} training pictures so far."
            )
        box = plate_box(self.stage)
        if box is None:
            return
        shown = fit_frame(visible, *box)
        self._photo = bgr_image(shown)
        self.view.configure(image=self._photo)

    def _save_still(self, raw, kind: str) -> Path:
        folder = DATASET_DIR / kind
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
        cv2.imwrite(str(path), raw)
        return path

    def _open_writer(self, frame) -> None:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = VIDEO_PREFIX[self.capture_label]
        path = VIDEO_DIR / f"{prefix}_{stamp}.mp4"
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height))
        if not writer.isOpened():
            path = path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), VIDEO_FPS, (width, height))
        if not writer.isOpened():
            self._append("Could not create a video file.")
            self.phase = "idle"
            self.status.configure(text="Could not create a video file.")
            self._style_capture_buttons()
            return
        self.writer = writer
        self.video_path = path
        self.frames_written = 0
        self.stills_written = 0
        self.last_video_frame = None
        self.last_still = None
        self.phase = "video_record"
        self._append(f"Recording {self._label_name(self.capture_label)}. Press Stop video when you are done.")
        self._style_capture_buttons()
        self.writer.write(frame)
        self.frames_written = 1
        self._save_still(frame, self.capture_label)
        self.stills_written = 1
        self.last_video_frame = time.monotonic()
        self.last_still = self.last_video_frame
        self._refresh()

    def _stop_video(self) -> None:
        writer = self.writer
        path = self.video_path
        frames = self.frames_written
        stills = self.stills_written
        label = self._label_name(self.capture_label)
        self.writer = None
        self.video_path = None
        self.frames_written = 0
        self.stills_written = 0
        if writer is not None:
            writer.release()
        if path is not None and frames < 2:
            path.unlink(missing_ok=True)
            path = None
        self.phase = "idle"
        if path is None:
            message = "Video was too short and was discarded."
        else:
            message = f"Saved {label} video {path.name} and {stills} training pictures."
        self.status.configure(text=message)
        self._append(message)
        self._refresh()
        self._style_capture_buttons()

    def shutdown(self) -> None:
        self._closed = True
        if self.phase == "video_record":
            self._stop_video()

    def _refresh(self) -> None:
        found = collect_samples()
        person = len(found["person"])
        unfeasible = len(found["unfeasible"])
        empty = len(found["no_person"])
        self.counts.configure(
            text=(
                f"Person pictures: {person}     Unfeasible pictures: {unfeasible}     "
                f"Empty pictures: {empty}     Need {MIN_PER_CLASS} of each"
            )
        )
        if MODEL_PATH.exists():
            self.model_label.configure(text=f"Current model: {MODEL_PATH}")
        else:
            self.model_label.configure(text="No trained model yet.")

    def _append(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if self._training or self.phase != "idle":
            return
        self._training = True
        self._style_capture_buttons()
        self._refresh()
        self._append("Starting.")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            train(log=lambda message: self.after(0, self._append, message))
        except Exception as exc:
            self.after(0, self._append, f"Training failed: {exc}")
        self.after(0, self._finish)

    def _finish(self) -> None:
        self._training = False
        self._style_capture_buttons()
        self._refresh()


def main() -> None:
    try:
        from kinect_app import run

        run("train")
    except Exception as exc:
        from setup_runtime import close_splash

        close_splash()
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Kinect", f"The window could not open.\n\n{exc}")
        root.destroy()


if __name__ == "__main__":
    main()
