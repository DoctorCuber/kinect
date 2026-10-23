"""Liquid glass. Light panels are frosted over a pale blue wash; dark panels are the same glass in deep blue."""

import tkinter as tk

import customtkinter as ctk

from settings_store import LIMITS, Settings, load, save

WINDOW = ("#E7F1FA", "#0E2240")
GLASS = ("#E7F0F8", "#17324F")
GLASS_EDGE = ("#FFFFFF", "#9EC0E4")
GLASS_RIM = ("#C5D9EE", "#2C5C90")
CARD = ("#F7FBFF", "#12283F")
CHIP = ("#E3EEF8", "#1A406C")
CHIP_HOVER = ("#F7FBFF", "#24548A")
TEXT = ("#1D1D1F", "#F5F5F7")
SECONDARY = ("#3A3A3C", "#D5DEEA")
ACCENT = ("#0071E3", "#0071E3")
ACCENT_TEXT = ("#FFFFFF", "#FFFFFF")
ACCENT_HOVER = ("#0060C4", "#005BB5")
SURFACE = ("#F6FAFE", "#13283F")
WELL = ("#14181E", "#0B1016")
WELL_TEXT = ("#F5F5F7", "#F5F5F7")
RADIUS = 22

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


def glass_frame(master, **kwargs):
    """Frosted plate: blurred wash, translucent tint, specular edge, and a soft shadow."""
    radius = kwargs.pop("corner_radius", RADIUS)
    even = kwargs.pop("even", False)
    kwargs.pop("fg_color", None)
    kwargs.pop("border_width", None)
    kwargs.pop("border_color", None)
    return GlassPanel(master, radius=radius, even=even, **kwargs)


def wash_frame(master, **kwargs):
    """Window wash, so the page around the plates is the same luminous field."""
    kwargs.pop("fg_color", None)
    return WashFrame(master, **kwargs)


def _dark_mode(mode=None) -> bool:
    if mode in ("Dark", "Light"):
        return mode == "Dark"
    return ctk.get_appearance_mode() == "Dark"


def _prefer_solid() -> bool:
    """Windows reduced-transparency. Plates stay milky, with the rim still drawn."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "EnableTransparency")
        return int(value) == 0
    except OSError:
        return False


def _frost_bitmap(width: int, height: int, dark: bool = False):
    """Luminous field the plates frost. Orbs stay off the top-left, where titles sit."""
    from PIL import Image, ImageDraw, ImageFilter

    if dark:
        base, veil = "#0E2240", "#102844"
        spots = (
            ("#1E4B86", (0.45, -0.18, 1.2, 0.42)),
            ("#16345A", (-0.15, 0.55, 0.6, 1.15)),
            ("#081628", (0.62, 0.62, 1.25, 1.2)),
        )
        blend = 0.22
    else:
        base, veil = "#E4EEF8", "#F4F8FC"
        spots = (
            ("#8EBFEE", (0.5, -0.2, 1.25, 0.48)),
            ("#C5DFF6", (-0.2, 0.5, 0.62, 1.15)),
            ("#7EB4EA", (0.72, 0.58, 1.3, 1.2)),
        )
        blend = 0.08
    picture = Image.new("RGB", (max(1, width), max(1, height)), base)
    draw = ImageDraw.Draw(picture)
    for color, (x0, y0, x1, y1) in spots:
        box = tuple(
            value if isinstance(value, int) else int((width if index % 2 == 0 else height) * value)
            for index, value in enumerate((x0, y0, x1, y1))
        )
        draw.ellipse(box, fill=color)
    picture = picture.filter(ImageFilter.GaussianBlur(radius=max(18, min(width, height) // 18)))
    return Image.blend(picture, Image.new("RGB", picture.size, veil), blend)


def _backdrop_crop(widget, width: int, height: int):
    from PIL import Image

    window = widget.winfo_toplevel()
    source = getattr(window, "_frost_pil", None)
    dark = _dark_mode()
    if source is None:
        return _frost_bitmap(width, height, dark)
    x = widget.winfo_rootx() - window.winfo_rootx()
    y = widget.winfo_rooty() - window.winfo_rooty()
    sw, sh = source.size
    left, top = max(0, x), max(0, y)
    right, bottom = min(sw, x + width), min(sh, y + height)
    if right <= left or bottom <= top:
        return _frost_bitmap(width, height, dark)
    crop = source.crop((left, top, right, bottom))
    if crop.size == (width, height):
        return crop
    canvas = Image.new("RGB", (width, height), "#0E2240" if dark else "#E4EEF8")
    canvas.paste(crop, (max(0, left - x), max(0, top - y)))
    return canvas


def _render_plate(crop, radius: int, dark: bool):
    """Frost the wash, then lay on a highlight, a reflection, and a rim."""
    from PIL import Image, ImageDraw, ImageFilter

    width, height = crop.size
    small = crop.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BILINEAR)
    blurred = small.filter(ImageFilter.GaussianBlur(radius=16)).resize((width, height), Image.Resampling.BILINEAR)
    veil = Image.new("RGB", (width, height), "#102844" if dark else "#F4F8FC")
    tint = 0.82 if _prefer_solid() else (0.46 if dark else 0.22)
    glass = Image.blend(blurred, veil, tint).convert("RGBA")

    sheen = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(sheen)
    reach = max(1, int(height * 0.22))
    for row in range(reach):
        alpha = int((64 if not dark else 42) * (1 - row / reach))
        pen.line([(0, row), (width, row)], fill=(255, 255, 255, alpha))
    glass = Image.alpha_composite(glass, sheen)

    streak = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(streak)
    pen.ellipse(
        (int(-width * 0.08), int(-height * 0.42), int(width * 0.78), int(height * 0.28)),
        fill=(255, 255, 255, 70 if not dark else 40),
    )
    glass = Image.alpha_composite(glass, streak.filter(ImageFilter.GaussianBlur(radius=22)))

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(shadow)
    # Keep the blur inside the bitmap. A shadow clipped by the square edge reads as a hard box.
    inset = 16
    if width > inset * 2 + 8 and height > inset * 2 + 8:
        pen.rounded_rectangle(
            (inset, inset + 4, width - inset, height - inset),
            radius=max(4, radius - 8),
            fill=(10, 24, 48, 70 if not dark else 110),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    base = Image.alpha_composite(crop.convert("RGBA"), shadow)

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((1, 1, width - 2, height - 2), radius=radius, fill=255)
    plate = Image.composite(glass, base, mask)

    rim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(rim)
    pen.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=radius,
        outline=(255, 255, 255, 210) if not dark else (168, 204, 236, 150),
        width=2,
    )
    pen.arc((3, 3, width - 4, max(8, int(height * 0.45))), 200, 340, fill=(255, 255, 255, 230), width=2)
    return Image.alpha_composite(plate, rim).convert("RGB")


def _render_even(crop, radius: int, dark: bool):
    """A flat field with the same rounded rim, so a list sitting on it has no second edge."""
    from PIL import Image, ImageDraw

    width, height = crop.size
    fill = (19, 40, 63, 255) if dark else (246, 250, 254, 255)
    color = Image.new("RGBA", (width, height), fill)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((1, 1, width - 2, height - 2), radius=radius, fill=255)
    plate = Image.composite(color, crop.convert("RGBA"), mask)
    rim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(rim)
    pen.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=radius,
        outline=(255, 255, 255, 210) if not dark else (168, 204, 236, 150),
        width=2,
    )
    return Image.alpha_composite(plate, rim).convert("RGB")


def _publish_frost(window, picture) -> None:
    window._frost_pil = picture
    for plate in list(getattr(window, "_glass_plates", ())):
        try:
            if plate.winfo_exists():
                plate.redraw(force=True)
        except tk.TclError:
            continue


class _LivePlate(ctk.CTkFrame):
    """A frame whose pixels are a picture sampled from the window wash."""

    def __init__(self, master, fg, painter, radius: int = 0, **kwargs) -> None:
        super().__init__(master, fg_color=fg, corner_radius=0, border_width=0, **kwargs)
        self._painter = painter
        self._radius = radius
        self._photo = None
        self._job = None
        self._last = None
        self._pending_force = False
        self._plate = ctk.CTkLabel(self, text="", fg_color=fg)
        self._plate.place(x=0, y=0, relwidth=1, relheight=1)
        top = self.winfo_toplevel()
        plates = getattr(top, "_glass_plates", None)
        if plates is None:
            top._glass_plates = plates = []
        plates.append(self)
        tk.Frame.bind(self, "<Configure>", self._on_configure, add="+")
        self.after(70, lambda: self.redraw(force=True))

    def _on_configure(self, _event) -> None:
        self.redraw()

    def redraw(self, force: bool = False) -> None:
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
        self._pending_force = force or self._pending_force
        self._job = self.after(40, self._paint)

    def _restack(self) -> None:
        self._plate.lift()
        for child in self.winfo_children():
            if child is not self._plate:
                child.lift()

    def _paint(self) -> None:
        self._job = None
        try:
            if not self.winfo_exists():
                return
            width, height = self.winfo_width(), self.winfo_height()
        except tk.TclError:
            return
        if width < 8 or height < 8:
            return
        mode = ctk.get_appearance_mode()
        force = self._pending_force
        self._pending_force = False
        last = self._last
        if (
            not force
            and last is not None
            and last[0] == mode
            and abs(width - last[1]) < 24
            and abs(height - last[2]) < 24
        ):
            self._restack()
            return
        self._last = (mode, width, height)
        picture = self._painter(_backdrop_crop(self, width, height), self._radius, mode == "Dark")
        self._picture = picture
        self._photo = ctk.CTkImage(light_image=picture, dark_image=picture, size=(width, height))
        self._plate.configure(image=self._photo)
        self._restack()
        _adopt_children(self)


def _hex(pixel) -> str:
    red, green, blue = pixel[:3]
    return f"#{red:02x}{green:02x}{blue:02x}"


def _plate_of(widget):
    parent = getattr(widget, "master", None)
    while parent is not None:
        if isinstance(parent, _LivePlate) and getattr(parent, "_picture", None) is not None:
            return parent
        parent = getattr(parent, "master", None)
    return None


def _crop_under(widget):
    """The plate pixels directly behind a widget."""
    plate = _plate_of(widget)
    if plate is None:
        return None
    picture = plate._picture
    width, height = widget.winfo_width(), widget.winfo_height()
    if width < 2 or height < 2:
        return None
    left = widget.winfo_rootx() - plate.winfo_rootx()
    top = widget.winfo_rooty() - plate.winfo_rooty()
    plate_width, plate_height = picture.size
    canvas = Image_new(picture, width, height)
    right = min(plate_width, left + width)
    bottom = min(plate_height, top + height)
    if right <= max(0, left) or bottom <= max(0, top):
        return canvas
    crop = picture.crop((max(0, left), max(0, top), right, bottom))
    canvas.paste(crop, (max(0, -left), max(0, -top)))
    return canvas


def Image_new(picture, width, height):
    from PIL import Image

    fill = picture.getpixel((0, 0))
    return Image.new("RGB", (width, height), fill)


def _mate(widget) -> None:
    """Keep a rounded control's square corners the same pixels as the glass behind it."""
    if getattr(widget, "_corner_mated", False):
        return
    widget._corner_mated = True
    original = widget._draw

    def wrapped(no_color_updates=False, _widget=widget, _original=original):
        _original(no_color_updates)
        _paint_mate(_widget)

    widget._draw = wrapped
    tk.Frame.bind(widget, "<Configure>", lambda _event, _widget=widget: _widget.after(30, lambda: _resample(_widget)), add="+")


def _resample(widget) -> None:
    try:
        if not widget.winfo_exists():
            return
    except tk.TclError:
        return
    radius = int(getattr(widget, "_corner_radius", 0) or 0)
    fg = getattr(widget, "_fg_color", None)
    transparent = fg in (None, "transparent")
    segmented_widget = isinstance(widget, ctk.CTkSegmentedButton)
    if not transparent and radius <= 0 and not segmented_widget:
        return
    crop = _crop_under(widget)
    if crop is None:
        return
    width, height = crop.size
    corners = (
        _hex(crop.getpixel((0, 0))),
        _hex(crop.getpixel((width - 1, 0))),
        _hex(crop.getpixel((width - 1, height - 1))),
        _hex(crop.getpixel((0, height - 1))),
    )
    changed = corners != getattr(widget, "_mate_colors", None) or (width, height) != getattr(widget, "_mate_size", None)
    widget._mate_colors = corners
    widget._mate_size = (width, height)
    widget._mate_ink = _hex(crop.getpixel((0, max(0, height // 2 - 1))))
    if transparent:
        from PIL import ImageTk

        widget._mate_photo = ImageTk.PhotoImage(crop)
    if segmented_widget and changed:
        widget.configure(background_corner_colors=corners)
    _paint_mate(widget)


def _paint_mate(widget) -> None:
    colors = getattr(widget, "_mate_colors", None)
    canvas = getattr(widget, "_canvas", None)
    if not colors or canvas is None:
        return
    try:
        if not canvas.winfo_exists():
            return
    except tk.TclError:
        return
    width, height = canvas.winfo_width(), canvas.winfo_height()
    if width < 2 or height < 2:
        return
    fg = getattr(widget, "_fg_color", None)
    transparent = fg in (None, "transparent")
    photo = getattr(widget, "_mate_photo", None)
    if transparent and photo is not None:
        canvas.delete("mate_image")
        canvas.create_image(0, 0, image=photo, anchor="nw", tags="mate_image")
        canvas.tag_raise("mate_image")
        if canvas.find_withtag("border_parts"):
            canvas.tag_raise("border_parts")
        canvas.configure(bg=colors[0])
        ink = getattr(widget, "_mate_ink", colors[0])
        for label in (getattr(widget, "_label", None), getattr(widget, "_text_label", None)):
            if label is None:
                continue
            try:
                label.configure(image=photo, compound="center", bg=ink)
            except tk.TclError:
                try:
                    label.configure(bg=ink)
                except tk.TclError:
                    pass
        return
    engine = getattr(widget, "_draw_engine", None)
    if engine is None or int(getattr(widget, "_corner_radius", 0) or 0) <= 0:
        return
    engine.draw_background_corners(width, height)
    for name, color in zip(
        (
            "background_corner_top_left",
            "background_corner_top_right",
            "background_corner_bottom_right",
            "background_corner_bottom_left",
        ),
        colors,
    ):
        canvas.itemconfig(name, fill=color)
    canvas.tag_lower("background_parts")
    canvas.configure(bg=colors[0])
    label = getattr(widget, "_text_label", None)
    if label is not None:
        try:
            label.configure(borderwidth=0, highlightthickness=0)
        except tk.TclError:
            pass


def _resample_native(widget) -> None:
    crop = _crop_under(widget)
    if crop is None:
        return
    color = _hex(crop.getpixel((crop.width // 2, crop.height // 2)))
    if getattr(widget, "_mate_bg", None) == color:
        return
    widget._mate_bg = color
    try:
        widget.configure(bg=color, highlightthickness=0, relief="flat", bd=0)
    except tk.TclError:
        try:
            widget.configure(bg=color)
        except tk.TclError:
            return


def _adopt_children(plate) -> None:
    def walk(widget) -> None:
        for child in widget.winfo_children():
            if child is getattr(widget, "_plate", None) or isinstance(child, _LivePlate):
                continue
            if isinstance(child, (tk.Listbox, tk.Spinbox)):
                _resample_native(child)
                continue
            if isinstance(child, ctk.CTkSegmentedButton):
                _mate(child)
                _resample(child)
                continue
            if hasattr(child, "_canvas") and hasattr(child, "_draw"):
                _mate(child)
                _resample(child)
            if isinstance(child, ctk.CTkFrame) and not isinstance(child, ctk.CTkSegmentedButton):
                walk(child)

    walk(plate)


class WashFrame(_LivePlate):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, WINDOW, lambda crop, _radius, _dark: crop, **kwargs)


class GlassPanel(_LivePlate):
    def __init__(self, master, radius: int = RADIUS, even: bool = False, **kwargs) -> None:
        painter = _render_even if even else _render_plate
        super().__init__(master, SURFACE if even else GLASS, painter, radius=radius, **kwargs)


def mount_frost(window) -> None:
    """Lay a blurred wash behind the window and ask Windows for acrylic on the frame."""
    holder = {"label": None, "image": None, "size": (0, 0), "mode": None}

    def redraw(_event=None, force: bool = False) -> None:
        width = max(int(window.winfo_width()), 480)
        height = max(int(window.winfo_height()), 360)
        mode = ctk.get_appearance_mode()
        previous = holder["size"]
        if (
            not force
            and holder["label"] is not None
            and holder["mode"] == mode
            and abs(width - previous[0]) < 48
            and abs(height - previous[1]) < 48
        ):
            holder["label"].lower()
            return
        holder["size"] = (width, height)
        holder["mode"] = mode
        picture = _frost_bitmap(width, height, dark=(mode == "Dark"))
        _publish_frost(window, picture)
        image = ctk.CTkImage(light_image=picture, dark_image=picture, size=(width, height))
        holder["image"] = image
        if holder["label"] is None:
            holder["label"] = ctk.CTkLabel(window, text="", image=image, fg_color="transparent")
            holder["label"].place(x=0, y=0, relwidth=1, relheight=1)
        else:
            holder["label"].configure(image=image)
        holder["label"].lower()

    window.bind("<Configure>", redraw, add="+")
    window._redraw_frost = redraw
    window.after(30, redraw)
    window.after(80, lambda: _acrylic(window))
    window._frost = holder
    ctk.AppearanceModeTracker.add(lambda _mode: redraw(force=True), window)


def _acrylic(window) -> None:
    try:
        import ctypes

        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        backdrop = ctypes.c_int(3)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
        dark = ctk.get_appearance_mode() == "Dark"
        caption = ctypes.c_int(0x00361E0C if dark else 0x00F8F4E7)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption))
        border = ctypes.c_int(0x00E4C09E if dark else 0x00F8DFC5)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(border), ctypes.sizeof(border))
    except Exception:
        return


class GlassSegment(ctk.CTkSegmentedButton):
    """Blue selected segment with white type, white segments with blue type."""

    def _select_button_by_value(self, value: str):
        super()._select_button_by_value(value)
        button = self._buttons_dict.get(value)
        if button is not None:
            button.configure(text_color=ACCENT_TEXT)

    def _unselect_button_by_value(self, value: str):
        super()._unselect_button_by_value(value)
        button = self._buttons_dict.get(value)
        if button is not None:
            button.configure(text_color=TEXT)


def segmented(master, values, command, **kwargs):
    """Segmented control: frosted track, blue selected pill."""
    return GlassSegment(
        master,
        values=values,
        command=command,
        corner_radius=16,
        fg_color=("#F8FBFE", "#1A3E68"),
        selected_color=ACCENT,
        selected_hover_color=ACCENT_HOVER,
        unselected_color=("#F8FBFE", "#1A3E68"),
        unselected_hover_color=CHIP_HOVER,
        text_color=TEXT,
        font=body_font(13),
        border_width=1,
        **kwargs,
    )


def apply_theme(mode: str, persist: bool = True) -> None:
    """Switch the glass between the pale look and the deep-blue look."""
    chosen = "dark" if str(mode).lower() == "dark" else "light"
    ctk.set_appearance_mode(chosen)
    if persist:
        settings = load()
        if settings.appearance != chosen:
            settings.appearance = chosen
            save(settings)


def saved_theme() -> str:
    return load().appearance if load().appearance in {"light", "dark"} else "light"


def apply_list(widget) -> None:
    """Tk list boxes cannot take color tuples, so paint them when the system appearance changes."""

    def paint(mode=None) -> None:
        if _dark_mode(mode):
            colors = {"bg": "#13283F", "fg": "#F5F5F7"}
        else:
            colors = {"bg": "#F6FAFE", "fg": "#1D1D1F"}
        widget.configure(
            selectbackground="#0071E3",
            selectforeground="#FFFFFF",
            highlightthickness=0,
            relief="flat",
            borderwidth=0,
            **colors,
        )

    paint()
    ctk.AppearanceModeTracker.add(paint, widget)


class AppButton(ctk.CTkButton):
    """Text button with a hover state. One accent button per view; the rest stay quiet."""

    def __init__(self, master, text="", command=None, kind="secondary", **kwargs) -> None:
        if kind == "primary":
            colors = {"fg_color": ACCENT, "text_color": ACCENT_TEXT, "hover_color": ACCENT_HOVER}
        else:
            colors = {"fg_color": CHIP, "text_color": TEXT, "hover_color": CHIP_HOVER, "border_width": 1, "border_color": GLASS_EDGE}
        colors.update(kwargs)
        super().__init__(
            master,
            text=text,
            command=command,
            corner_radius=16,
            height=32,
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


def _style_spinbox(box) -> None:
    def paint(mode=None) -> None:
        if _dark_mode(mode):
            colors = {"bg": "#13283F", "fg": "#F5F5F7", "buttonbackground": "#1A406C", "highlightbackground": "#2C5C90"}
        else:
            colors = {"bg": "#F6FAFE", "fg": "#1D1D1F", "buttonbackground": "#E3EEF8", "highlightbackground": "#C5D9EE"}
        box.configure(relief="flat", highlightthickness=0, bd=0, highlightcolor="#0071E3", **colors)

    paint()
    ctk.AppearanceModeTracker.add(paint, box)


def open_settings(parent, on_saved=None) -> None:
    current = load()
    window = ctk.CTkToplevel(parent)
    window.title("Settings")
    window.configure(fg_color=WINDOW)
    window.geometry("560x620")
    window.minsize(480, 520)
    window.transient(parent)
    window.grab_set()
    mount_frost(window)

    shell = wash_frame(window)
    shell.pack(fill="both", expand=True, padx=24, pady=20)
    ctk.CTkLabel(shell, text="Settings", text_color=TEXT, font=body_font(22, "bold"), anchor="w").pack(anchor="w")
    ctk.CTkLabel(
        shell,
        text="Grace and the clip buffers apply the next time you press Start. The shutter countdown applies to the next photo or video.",
        text_color=SECONDARY,
        font=body_font(13),
        wraplength=500,
        justify="left",
        anchor="w",
    ).pack(anchor="w", pady=(4, 16))

    fields = glass_frame(shell, corner_radius=18)
    fields.pack(fill="x", pady=(0, 8))
    variables = {}
    spins = []
    for key, title, unit, step in FIELDS:
        low, high = LIMITS[key]
        row = ctk.CTkFrame(fields, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=8)
        ctk.CTkLabel(row, text=f"{title} ({unit})", text_color=TEXT, font=body_font(13), anchor="w").pack(side="left")
        variable = tk.DoubleVar(value=getattr(current, key))
        variables[key] = variable
        holder = ctk.CTkFrame(row, fg_color=SURFACE, corner_radius=10)
        holder.pack(side="right")
        box = tk.Spinbox(
            holder,
            from_=low,
            to=high,
            increment=step,
            textvariable=variable,
            width=8,
            font=("Segoe UI", 13),
            justify="right",
            relief="flat",
            highlightthickness=0,
            bd=0,
        )
        box.pack(padx=10, pady=4)
        _style_spinbox(box)
        spins.append(box)

    buttons = wash_frame(shell)
    buttons.pack(fill="x", pady=(16, 0))

    def _save() -> None:
        updated = Settings()
        updated.appearance = "dark" if ctk.get_appearance_mode() == "Dark" else "light"
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
