"""Browse scored clips and export the set as a PDF or a CSV."""

import csv
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

import cv2

from ui_kit import SECONDARY, TEXT, WELL, WELL_TEXT, WINDOW, AppButton, apply_list, bgr_image, body_font, fit_frame, glass_frame, mount_frost, plate_box, wash_frame
from person_check import class_probabilities, clear_clip_history, frame_call, load_model

# Below this percent, the report says the room is empty.
# At this percent or higher, it says someone is here.
PRESENCE_EMPTY_BELOW = 15

LABELS = {"person": "Person", "no_person": "No person", "unfeasible": "Unfeasible"}


def person_presence(result: dict) -> float:
    """How sure the system is that a person is in the room, from 0 to 1.

    New scores store this as ``presence`` (one minus the chance the room is empty).
    Older scores only stored the winning label, so the same number is rebuilt from that.
    """
    if result.get("presence") is not None:
        value = float(result["presence"])
    else:
        label = result.get("image_label", result.get("label"))
        confidence = float(result.get("image_confidence", result.get("confidence", 0.0)))
        if label == "no_person":
            value = 1.0 - confidence
        else:
            value = confidence
    return min(1.0, max(0.0, value))


def presence_percent(result: dict) -> int:
    return int(round(person_presence(result) * 100))


def row_text(result: dict) -> str:
    percent = presence_percent(result)
    name = Path(result["clip"]).name
    note = "  (video only)" if result.get("source") == "video" else ""
    if percent < PRESENCE_EMPTY_BELOW:
        return f"Empty room   {name}{note}"
    return f"Someone here  {percent}%   {name}{note}"


def _open_video(path: Path):
    """Open a saved recording inside this app. Try the decoders that can read the Kinect mp4."""
    if not path.exists():
        return None
    apis = []
    for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_ANY"):
        api = getattr(cv2, name, None)
        if api is not None and api not in apis:
            apis.append(api)
    for api in apis:
        capture = cv2.VideoCapture(str(path), api)
        if not capture.isOpened():
            capture.release()
            continue
        ok, _frame = capture.read()
        if not ok:
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return capture
    return None


def _still(path: Path):
    capture = _open_video(path)
    if capture is None:
        return None
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 2:
        capture.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def describe(result: dict) -> str:
    """Plain-language explanation of one saved walk clip."""
    percent = presence_percent(result)
    name = Path(result["clip"]).name
    if percent < PRESENCE_EMPTY_BELOW:
        summary = "The room looks empty."
    else:
        summary = f"{percent}% sure that someone is here."
    lines = [summary, name]
    pose = result.get("pose")
    someone_here = percent >= PRESENCE_EMPTY_BELOW
    if pose == "lying":
        lines.append("The body is stretched out flat, the way someone looks when they are lying in bed.")
    elif pose == "cloud":
        lines.append("The joints collapsed into a small cluster, not a full person.")
    elif pose == "proportions":
        lines.append("The arms or legs are the wrong length for a real person.")
    elif someone_here and result.get("image_label") == "unfeasible":
        lines.append("The picture looks like a person who is lying down, not standing.")
    elif someone_here and result.get("label") == "person":
        lines.append("The shape of the body fits someone who is standing.")
    if result.get("source") == "video":
        lines.append("The separate infrared stills were missing, so this reading comes from the video and is only a rough one.")
    return "\n".join(lines)


class ReportWindow(ctk.CTkToplevel):
    def __init__(self, parent, results: list[dict], on_cleared=None) -> None:
        super().__init__(parent)
        self.title("Sleepwalk Report")
        self.configure(fg_color=WINDOW)
        self.geometry("980x680")
        self.minsize(860, 560)
        self.results = list(results)
        self.on_cleared = on_cleared
        self.index = 0
        self.capture = None
        self._photo = None
        self.playing = False
        self._ended = False
        self._frame_delay = 80
        self._closed = False
        self._play_index = 0
        self._shown_index = None
        self._score_generation = 0
        self._frame_calls: list[tuple[str, float, bool] | None] = []
        self._person_seen = False
        self._scores_complete = False
        self._score_error = ""
        self._model_pack = None
        self._model_lock = threading.Lock()
        self._score_lock = threading.Lock()

        shell = wash_frame(self)
        shell.pack(fill="both", expand=True, padx=24, pady=20)
        ctk.CTkLabel(shell, text="Sleepwalk Report", text_color=TEXT, font=body_font(22, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(
            shell,
            text="Each row is a clip saved when someone walked through the view. Play recording watches that video in this window and shows what the model calls the frame on screen. An empty frame after someone leaves does not hide that a person was in the recording. Under 15%, the saved score says the room looks empty. At 15% or more, it says that percent sure someone is here.",
            text_color=SECONDARY,
            font=body_font(13),
            wraplength=920,
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(2, 10))

        body = wash_frame(shell)
        body.pack(fill="both", expand=True)
        list_well = glass_frame(body, corner_radius=18, even=True)
        list_well.pack(side="left", fill="y")
        self.list = tk.Listbox(list_well, width=42, highlightthickness=0, relief="flat", font=("Segoe UI", 12), activestyle="none")
        apply_list(self.list)
        self.list.pack(fill="both", expand=True, padx=20, pady=16)
        self.list.bind("<<ListboxSelect>>", self._on_select)
        self.list.bind("<Double-Button-1>", self._play_row)
        for result in self.results:
            self.list.insert("end", row_text(result))

        right = wash_frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.stage = glass_frame(right, corner_radius=28)
        self.stage.pack(fill="both", expand=True)
        self.stage.pack_propagate(False)
        self.view = ctk.CTkLabel(self.stage, text="Select a recording to watch it here.", text_color=WELL_TEXT, fg_color=WELL, corner_radius=16)
        self.view.pack(expand=True, padx=12, pady=12)
        self._pending_frame = None
        self._paint_wait = False
        self.stage.bind("<Configure>", lambda _event: self._paint_pending())
        self.detail = ctk.CTkLabel(right, text="", text_color=TEXT, font=body_font(13), justify="left", anchor="w", wraplength=640)
        self.detail.pack(fill="x", pady=(8, 0))

        controls = wash_frame(shell)
        controls.pack(fill="x", pady=(12, 0))
        self.play_button = AppButton(controls, text="Play recording", command=self._toggle_play, kind="primary")
        self.play_button.pack(side="left")
        self.previous_button = AppButton(controls, text="Previous", command=self._previous)
        self.previous_button.pack(side="left", padx=(8, 0))
        self.next_button = AppButton(controls, text="Next", command=self._next)
        self.next_button.pack(side="left", padx=(8, 0))
        self.clear_button = AppButton(controls, text="Clear History", command=self._clear_history)
        self.clear_button.pack(side="left", padx=(8, 0))
        self.pdf_button = AppButton(controls, text="Export PDF", command=self._export_pdf)
        self.pdf_button.pack(side="right")
        self.csv_button = AppButton(controls, text="Export CSV", command=self._export_csv)
        self.csv_button.pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._close)
        mount_frost(self)
        if self.results:
            self.list.selection_set(0)
            self._show(0)
        self.after(20, self._take_focus)
        self.after(300, self._take_focus)

    def _take_focus(self) -> None:
        if self._closed:
            return
        owner = self.master.winfo_toplevel()
        try:
            owner.attributes("-topmost", False)
        except Exception:
            owner = None
        self.deiconify()
        if owner is not None:
            self.lift(owner)
        else:
            self.lift()
        self.focus_force()
        try:
            self.attributes("-topmost", True)
            self.after(200, self._drop_topmost)
            import ctypes

            handle = ctypes.windll.user32.GetParent(self.winfo_id())
            ctypes.windll.user32.SetForegroundWindow(handle)
        except Exception:
            return

    def _drop_topmost(self) -> None:
        if self._closed:
            return
        try:
            self.attributes("-topmost", False)
            self.lift()
            self.focus_force()
        except Exception:
            return

    def _on_select(self, _event) -> None:
        selection = self.list.curselection()
        if selection and selection[0] < len(self.results):
            self._show(selection[0])

    def _play_row(self, _event) -> None:
        selection = self.list.curselection()
        if not selection or selection[0] >= len(self.results):
            return
        if selection[0] != self.index or self.capture is None:
            self._show(selection[0])
        if not self.playing:
            self._toggle_play()

    def _show(self, index: int) -> None:
        self._stop_playback()
        self.index = index
        result = self.results[index]
        self.list.selection_clear(0, "end")
        self.list.selection_set(index)
        self.list.see(index)
        path = Path(result["clip"])
        self._begin_scoring(path)
        self._open_clip(path)
        if self.capture is None:
            self._photo = None
            self._pending_frame = None
            self._shown_index = None
            self.view.configure(image=None, text="This recording could not be opened in the report.")
            self._update_detail()
            return
        ok, frame = self.capture.read()
        self._shown_index = 0 if ok else None
        self._play_index = 0
        if ok:
            self._paint_frame(frame)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._update_detail()

    def _paint_frame(self, frame) -> None:
        self._pending_frame = frame
        self._paint_pending()

    def _paint_pending(self) -> None:
        frame = self._pending_frame
        if frame is None or self._closed:
            return
        box = plate_box(self.stage)
        if box is None:
            if not self._paint_wait:
                self._paint_wait = True
                self.after(40, self._paint_later)
            return
        self._paint_wait = False
        shown = fit_frame(frame, *box)
        self._photo = bgr_image(shown)
        self.view.configure(image=self._photo, text="")

    def _paint_later(self) -> None:
        self._paint_wait = False
        self._paint_pending()

    def _open_clip(self, path: Path) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._ended = False
        capture = _open_video(path)
        if capture is None:
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 12)
        self._frame_delay = max(30, int(round(1000 / max(fps, 1))))
        self.capture = capture

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop_playback()
            return
        if not self.results:
            return
        path = Path(self.results[self.index]["clip"])
        if self.capture is None or not self.capture.isOpened() or self._ended:
            self._open_clip(path)
        if self.capture is None:
            self.view.configure(image=None, text="This recording could not be opened in the report.")
            self.detail.configure(text=describe(self.results[self.index]) + "\n\nThe saved recording could not be opened in the report.")
            return
        if self._ended:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._ended = False
            self._play_index = 0
        self.playing = True
        self.play_button.style(text="Pause")
        self._advance()

    def _advance(self) -> None:
        if self._closed or not self.playing or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self._ended = True
            self._stop_playback()
            self.play_button.style(text="Play recording")
            self._update_detail()
            return
        self._shown_index = self._play_index
        self._play_index += 1
        self._paint_frame(frame)
        self._update_detail()
        self.after(self._frame_delay, self._advance)

    def _begin_scoring(self, path: Path) -> None:
        """Classify every frame of this clip. A new clip cancels the previous pass."""
        self._score_generation += 1
        generation = self._score_generation
        with self._score_lock:
            self._frame_calls = []
            self._person_seen = False
            self._scores_complete = False
            self._score_error = ""
        self._shown_index = None
        self._play_index = 0
        threading.Thread(target=self._score_clip_frames, args=(path, generation), daemon=True).start()
        self.after(200, lambda: self._refresh_score_text(generation))

    def _score_clip_frames(self, path: Path, generation: int) -> None:
        try:
            model_pack = self._ensure_model()
        except FileNotFoundError:
            self._finish_scoring(generation, "No person model yet, so frames cannot be classified.")
            return
        except Exception as exc:
            self._finish_scoring(generation, f"Frame classification failed: {exc}")
            return
        capture = _open_video(path)
        if capture is None:
            self._finish_scoring(generation, "This recording could not be classified frame by frame.")
            return
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        with self._score_lock:
            if generation == self._score_generation:
                self._frame_calls = [None] * count if count > 0 else []
        index = 0
        try:
            while not self._closed and generation == self._score_generation:
                ok, frame = capture.read()
                if not ok:
                    break
                with self._model_lock:
                    probabilities = class_probabilities(*model_pack, frame)
                label, confidence, present = frame_call(probabilities)
                with self._score_lock:
                    if generation != self._score_generation:
                        break
                    if index >= len(self._frame_calls):
                        self._frame_calls.extend([None] * (index + 1 - len(self._frame_calls)))
                    self._frame_calls[index] = (label, confidence, present)
                    if present:
                        self._person_seen = True
                index += 1
        except Exception as exc:
            self._finish_scoring(generation, f"Frame classification failed: {exc}")
            return
        finally:
            capture.release()
        self._finish_scoring(generation, "")

    def _ensure_model(self):
        with self._model_lock:
            if self._model_pack is None:
                model, transform, classes, device = load_model()
                self._model_pack = (model, transform, classes, device)
            return self._model_pack

    def _finish_scoring(self, generation: int, error: str) -> None:
        with self._score_lock:
            if generation != self._score_generation:
                return
            if error:
                self._score_error = error
            self._scores_complete = True

    def _refresh_score_text(self, generation: int) -> None:
        if self._closed or generation != self._score_generation:
            return
        self._update_detail()
        with self._score_lock:
            done = self._scores_complete
        if not done:
            self.after(200, lambda: self._refresh_score_text(generation))

    def _update_detail(self) -> None:
        if self._closed or not self.results or self.index >= len(self.results):
            return
        result = self.results[self.index]
        with self._score_lock:
            call = None
            shown = self._shown_index
            if shown is not None and 0 <= shown < len(self._frame_calls):
                call = self._frame_calls[shown]
            person_seen = self._person_seen
            complete = self._scores_complete
            error = self._score_error
        if error:
            frame_line = error
        elif call is None and complete:
            frame_line = "This frame: no classification."
        elif call is None:
            frame_line = "This frame: scoring it now."
        else:
            label, confidence, _present = call
            name = LABELS.get(label, label)
            percent = int(round(confidence * 100))
            frame_line = f"This frame: {name} {percent}%."
        if person_seen:
            anywhere = "A person was detected in this recording."
        elif complete and not error:
            anywhere = "No person was detected in this recording."
        else:
            anywhere = "Still checking every frame for a person."
        self.detail.configure(text="\n".join((frame_line, anywhere, describe(result))))

    def _stop_playback(self) -> None:
        self.playing = False
        self.play_button.style(text="Play recording")

    def _previous(self) -> None:
        if self.results:
            self._show((self.index - 1) % len(self.results))

    def _next(self) -> None:
        if self.results:
            self._show((self.index + 1) % len(self.results))

    def _clear_history(self) -> None:
        if not self.results or not self._ask_clear():
            return
        self._stop_playback()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.on_cleared is not None:
            self.on_cleared()
        else:
            clear_clip_history()
        self._score_generation += 1
        self.results = []
        self._pending_frame = None
        self._shown_index = None
        self.list.delete(0, "end")
        self.list.insert("end", "No clips yet.")
        self.view.configure(image=None)
        self.detail.configure(text="History cleared. Training pictures and the person model were kept.")
        for button in (self.play_button, self.previous_button, self.next_button, self.clear_button, self.pdf_button, self.csv_button):
            button.configure(state="disabled")

    def _ask_clear(self) -> bool:
        dialog = ctk.CTkToplevel(self)
        dialog.title("Clear History")
        dialog.configure(fg_color=WINDOW)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        mount_frost(dialog)
        choice = {"ok": False}
        shell = wash_frame(dialog)
        shell.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(shell, text="Clear Report History?", text_color=TEXT, font=body_font(17, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(
            shell,
            text="Saved clips and their scores will be removed. Training pictures and the person model stay.",
            text_color=SECONDARY,
            font=body_font(13),
            wraplength=380,
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(8, 0))
        row = ctk.CTkFrame(shell, fg_color="transparent")
        row.pack(fill="x", pady=(16, 0))

        def confirm() -> None:
            choice["ok"] = True
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        AppButton(row, text="Clear History", command=confirm, kind="primary").pack(side="right")
        AppButton(row, text="Cancel", command=cancel).pack(side="right", padx=(0, 8))
        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.update_idletasks()
        width = max(dialog.winfo_reqwidth(), 440)
        height = dialog.winfo_reqheight()
        x = self.winfo_rootx() + max(0, (self.winfo_width() - width) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.wait_window()
        return choice["ok"]

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save report as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="sleepwalk_report.csv",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["clip", "label", "confidence", "image_label", "image_confidence", "pose", "frames", "source"])
            for result in self.results:
                writer.writerow(
                    [
                        Path(result["clip"]).name,
                        result.get("label", ""),
                        f"{float(result.get('confidence', 0)):.3f}",
                        result.get("image_label", ""),
                        f"{float(result.get('image_confidence', 0)):.3f}",
                        result.get("pose", ""),
                        result.get("frames", ""),
                        result.get("source", ""),
                    ]
                )
        messagebox.showinfo("Sleepwalk report", f"Saved {path}", parent=self)

    def _export_pdf(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save report as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile="sleepwalk_report.pdf",
        )
        if not path:
            return
        try:
            from fpdf import FPDF
        except ImportError as exc:
            messagebox.showerror(
                "Sleepwalk report",
                "Windows Application Control blocked the PDF library, so this report could not be saved as a PDF.\n"
                "CSV export still works.\n\n"
                f"{exc}",
                parent=self,
            )
            return
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=16)
        pdf.add_page()
        pdf.set_font("Helvetica", size=18)
        pdf.cell(0, 12, "Sleepwalk report", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 8, datetime.now().strftime("Created %Y-%m-%d %H:%M"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.multi_cell(
            0,
            6,
            "Under 15%, the report says the room looks empty. "
            "At 15% or more, it says that percent sure someone is here.",
        )
        folder = Path(path).with_suffix("")
        picture_dir = folder.parent / (folder.name + "_pages")
        picture_dir.mkdir(parents=True, exist_ok=True)
        for number, result in enumerate(self.results, start=1):
            pdf.ln(4)
            pdf.set_font("Helvetica", size=13)
            pdf.cell(0, 8, f"{number}. {Path(result['clip']).name}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, describe(result))
            frame = _still(Path(result["clip"]))
            if frame is None:
                continue
            picture = picture_dir / f"{number:02d}.jpg"
            cv2.imwrite(str(picture), frame)
            pdf.image(str(picture), w=120)
        pdf.output(path)
        messagebox.showinfo("Sleepwalk report", f"Saved {path}", parent=self)

    def _close(self) -> None:
        self._closed = True
        self._score_generation += 1
        self.playing = False
        if self.capture is not None:
            self.capture.release()
        self.destroy()
