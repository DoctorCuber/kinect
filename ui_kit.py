"""Apple-inspired tokens and controls for the Kinect window. Colors are (light, dark)."""

import tkinter as tk

import customtkinter as ctk

from settings_store import LIMITS, Settings, load, save

WINDOW = ("#F5F5F7", "#1C1C1E")
CARD = ("#FFFFFF", "#2C2C2E")
TEXT = ("#1D1D1F", "#F5F5F7")
SECONDARY = ("#3A3A3C", "#AEAEB2")
ACCENT = ("#0071E3", "#0071E3")
ACCENT_TEXT = ("#FFFFFF", "#FFFFFF")
ACCENT_HOVER = ("#0060C4", "#005BB5")
WELL = ("#0B0B0D", "#0B0B0D")

PILL = {
    "idle": (("#E5E5EA", "#3A3A3C"), ("#1D1D1F", "#F5F5F7")),
    "grace": (("#C93400", "#FF9F0A"), ("#FFFFFF", "#1D1D1F")),
    "watch": (("#1B7A32", "#30D158"), ("#FFFFFF", "#1D1D1F")),
    "rec": (("#D70015", "#FF453A"), ("#FFFFFF", "#1D1D1F")),
}

FIELDS = (
    ("grace_seconds", "Grace Period Before Monitoring", "seconds", 1),
    ("pre_seconds", "Seconds Saved Before a Walk", "seconds", 1),
    ("post_seconds", "Seconds Saved After the Person Leaves", "seconds", 1),
    ("walk_distance_meters", "Distance That Counts as a Walk", "meters", 0.1),
    ("shutter_seconds", "Trainer Shutter Countdown", "seconds", 1),
    ("min_body_span_meters", "Smallest Span That Still Counts as a Body", "meters", 0.05),
    ("lying_height_meters", "Vertical Span Below This Counts as Lying Down", "meters", 0.05),
)


def plate_box(widget, pad: int = 16) -> tuple[int, int] | None:
    """Inner size of a camera well, in the same units CustomTkinter uses for images."""
    width = widget.winfo_width()
    height = widget.winfo_height()
    if width < 80 or height < 80:
        return None
    try:
        scaling = float(ctk.ScalingTracker.get_widget_scaling(widget) or 1.0)
    except Exception:
        scaling = 1.0
    if scaling <= 0:
        scaling = 1.0
    return max(1, int(width / scaling) - pad), max(1, int(height / scaling) - pad)


def fit_frame(image, width: int, height: int):
    """Scale a frame to fit a box and keep its aspect ratio."""
    import cv2

    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (max(1, int(source_width * scale)), max(1, int(source_height * scale))),
        interpolation=interpolation,
    )


def bgr_image(frame):
    """BGR frame as a CustomTkinter image that stays sharp on scaled displays."""
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    picture = Image.fromarray(rgb)
    return ctk.CTkImage(light_image=picture, dark_image=picture, size=(picture.width, picture.height))


def body_font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


def segmented(master, values, command, **kwargs):
    """macOS-style segment: a raised chip on a quiet track, same label color either way."""
    return ctk.CTkSegmentedButton(
        master,
        values=values,
        command=command,
        corner_radius=8,
        fg_color=("#E5E5EA", "#3A3A3C"),
        selected_color=("#FFFFFF", "#636366"),
        selected_hover_color=("#F2F2F4", "#545458"),
        unselected_color=("#E5E5EA", "#3A3A3C"),
        unselected_hover_color=("#D2D2D7", "#48484A"),
        text_color=TEXT,
        font=body_font(13),
        **kwargs,
    )


def apply_list(widget) -> None:
    """Tk list boxes cannot take color tuples, so paint them when the system appearance changes."""

    def paint(mode=None) -> None:
        dark = mode == "Dark" if mode in ("Dark", "Light") else ctk.get_appearance_mode() == "Dark"
        if dark:
            widget.configure(bg="#2C2C2E", fg="#F5F5F7", selectbackground="#0071E3", selectforeground="#FFFFFF")
        else:
            widget.configure(bg="#FFFFFF", fg="#1D1D1F", selectbackground="#0071E3", selectforeground="#FFFFFF")

    paint()
    ctk.AppearanceModeTracker.add(paint, widget)


class AppButton(ctk.CTkButton):
    """Text button with a hover state. One accent button per view; the rest stay quiet."""

    def __init__(self, master, text="", command=None, kind="secondary", **kwargs) -> None:
        if kind == "primary":
            colors = {"fg_color": ACCENT, "text_color": ACCENT_TEXT, "hover_color": ACCENT_HOVER}
        else:
            colors = {"fg_color": CARD, "text_color": TEXT, "hover_color": ("#E8E8ED", "#3A3A3C"), "border_width": 1, "border_color": ("#D2D2D7", "#48484A")}
        colors.update(kwargs)
        super().__init__(
            master,
            text=text,
            command=command,
            corner_radius=8,
            height=28,
            font=body_font(13, "bold" if kind == "primary" else "normal"),
            **colors,
        )

    def style(self, **kwargs) -> None:
        self.configure(require_redraw=True, **kwargs)

    def configure(self, require_redraw=False, **kwargs):
        if "bg" in kwargs:
            kwargs["fg_color"] = kwargs.pop("bg")
        if "fg" in kwargs:
            kwargs["text_color"] = kwargs.pop("fg")
        return super().configure(require_redraw=require_redraw, **kwargs)


def open_settings(parent, on_saved=None) -> None:
    current = load()
    window = ctk.CTkToplevel(parent)
    window.title("Settings")
    window.configure(fg_color=WINDOW)
    window.geometry("560x620")
    window.minsize(480, 520)
    window.transient(parent)
    window.grab_set()

    shell = ctk.CTkFrame(window, fg_color="transparent")
    shell.pack(fill="both", expand=True, padx=20, pady=16)
    ctk.CTkLabel(shell, text="Settings", text_color=TEXT, font=body_font(22, "bold"), anchor="w").pack(anchor="w")
    ctk.CTkLabel(
        shell,
        text="Grace and the clip buffers apply the next time you press Start. The shutter countdown applies to the next photo or video.",
        text_color=SECONDARY,
        font=body_font(13),
        wraplength=500,
        justify="left",
        anchor="w",
    ).pack(anchor="w", pady=(4, 12))

    variables = {}
    for key, title, unit, step in FIELDS:
        low, high = LIMITS[key]
        row = ctk.CTkFrame(shell, fg_color="transparent")
        row.pack(fill="x", pady=6)
        ctk.CTkLabel(row, text=f"{title} ({unit})", text_color=TEXT, font=body_font(13), anchor="w").pack(side="left")
        variable = tk.DoubleVar(value=getattr(current, key))
        variables[key] = variable
        tk.Spinbox(
            row,
            from_=low,
            to=high,
            increment=step,
            textvariable=variable,
            width=8,
            font=("Segoe UI", 13),
            justify="right",
        ).pack(side="right")

    buttons = ctk.CTkFrame(shell, fg_color="transparent")
    buttons.pack(fill="x", pady=(16, 0))

    def _save() -> None:
        updated = Settings()
        for key, _title, _unit, _step in FIELDS:
            low, high = LIMITS[key]
            try:
                value = float(variables[key].get())
            except (tk.TclError, ValueError):
                value = getattr(current, key)
            setattr(updated, key, min(high, max(low, value)))
        save(updated)
        if on_saved is not None:
            on_saved(updated)
        window.destroy()

    AppButton(buttons, text="Save", command=_save, kind="primary").pack(side="left")
    AppButton(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=(8, 0))
