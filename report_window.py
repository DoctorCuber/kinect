"""Browse scored clips and export the set as a PDF or a CSV."""

import csv
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

import cv2
from fpdf import FPDF

from ui_kit import SECONDARY, TEXT, WINDOW, AppButton, apply_list, bgr_image, body_font, fit_frame, plate_box
from person_check import clear_clip_history, result_text
from pose_check import POSE_NOTES

LABELS = {"person": "Person", "no_person": "No person", "unfeasible": "Unfeasible"}


def _still(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 2:
        capture.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def describe(result: dict) -> str:
    label = LABELS.get(result.get("label"), result.get("label", ""))
    percent = int(round(float(result.get("confidence", 0)) * 100))
    image_label = LABELS.get(result.get("image_label", ""), "")
    lines = [f"{label}  {percent}%", Path(result["clip"]).name]
    if image_label:
        image_percent = int(round(float(result.get("image_confidence", 0)) * 100))
        lines.append(f"Infrared model: {image_label} {image_percent}%.")
    pose = result.get("pose")
    if pose in POSE_NOTES:
        lines.append(POSE_NOTES[pose])
    elif pose == "ok" and result.get("image_label") == "unfeasible":
        lines.append("The infrared model saw a person who is not standing, such as someone lying down.")
    elif pose == "ok":
        lines.append("The skeleton had the size and shape of an upright person.")
    if result.get("source") == "video":
        lines.append("Scored from the clip because raw infrared frames were missing.")
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
        self._closed = False

        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True, padx=20, pady=16)
        ctk.CTkLabel(shell, text="Sleepwalk Report", text_color=TEXT, font=body_font(22, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(
            shell,
            text="Person means a standing human. Unfeasible means lying down, a short joint cloud, or limb lengths that cannot be a person.",
            text_color=SECONDARY,
            font=body_font(13),
            wraplength=920,
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(2, 10))

        body = ctk.CTkFrame(shell, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self.list = tk.Listbox(body, width=42, highlightthickness=0, relief="flat", font=("Segoe UI", 12), activestyle="none")
        apply_list(self.list)
        self.list.pack(side="left", fill="y")
        self.list.bind("<<ListboxSelect>>", self._on_select)
        for result in self.results:
            self.list.insert("end", result_text(result))

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.stage = ctk.CTkFrame(right, fg_color="#0B0B0D", corner_radius=12)
        self.stage.pack(fill="both", expand=True)
        self.stage.pack_propagate(False)
        self.view = ctk.CTkLabel(self.stage, text="", fg_color="#0B0B0D")
        self.view.pack(expand=True, padx=8, pady=8)
        self._pending_frame = None
        self._paint_wait = False
        self.stage.bind("<Configure>", lambda _event: self._paint_pending())
        self.detail = ctk.CTkLabel(right, text="", text_color=TEXT, font=body_font(13), justify="left", anchor="w")
        self.detail.pack(fill="x", pady=(8, 0))

        controls = ctk.CTkFrame(shell, fg_color="transparent")
        controls.pack(fill="x", pady=(12, 0))
        self.play_button = AppButton(controls, text="Play", command=self._toggle_play, kind="primary")
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
        if self.results:
            self.list.selection_set(0)
            self._show(0)

    def _on_select(self, _event) -> None:
        selection = self.list.curselection()
        if selection:
            self._show(selection[0])

    def _show(self, index: int) -> None:
        self._stop_playback()
        self.index = index
        result = self.results[index]
        self.detail.configure(text=describe(result))
        self.list.selection_clear(0, "end")
        self.list.selection_set(index)
        self.list.see(index)
        path = Path(result["clip"])
        frame = _still(path)
        if frame is not None:
            self._paint_frame(frame)
        self.capture = cv2.VideoCapture(str(path)) if path.exists() else None

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
        self.view.configure(image=self._photo)

    def _paint_later(self) -> None:
        self._paint_wait = False
        self._paint_pending()

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop_playback()
            return
        if self.capture is None or not self.capture.isOpened():
            return
        self.playing = True
        self.play_button.style(text="Pause")
        self._advance()

    def _advance(self) -> None:
        if self._closed or not self.playing or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if ok:
            self._paint_frame(frame)
        self.after(80, self._advance)

    def _stop_playback(self) -> None:
        self.playing = False
        self.play_button.style(text="Play")

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
        self.results = []
        self._pending_frame = None
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
        choice = {"ok": False}
        ctk.CTkLabel(dialog, text="Clear Report History?", text_color=TEXT, font=body_font(17, "bold"), anchor="w").pack(anchor="w", padx=20, pady=(20, 8))
        ctk.CTkLabel(
            dialog,
            text="Saved clips and their scores will be removed. Training pictures and the person model stay.",
            text_color=SECONDARY,
            font=body_font(13),
            wraplength=380,
            justify="left",
            anchor="w",
        ).pack(anchor="w", padx=20)
        row = ctk.CTkFrame(dialog, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(16, 20))

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
            "Person means a standing human. "
            "No person means the infrared model did not see one. "
            "Unfeasible means someone lying down, a short joint cloud, or limb lengths that cannot be a person.",
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
        self.playing = False
        if self.capture is not None:
            self.capture.release()
        self.destroy()
