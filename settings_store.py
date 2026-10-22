"""Saved options shared by the detector and the trainer."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PATH = Path(__file__).resolve().parent / "settings.json"


@dataclass
class Settings:
    grace_seconds: float = 120
    pre_seconds: float = 10
    post_seconds: float = 10
    walk_distance_meters: float = 0.4
    shutter_seconds: float = 5
    min_body_span_meters: float = 0.55
    lying_height_meters: float = 0.45


LIMITS = {
    "grace_seconds": (0, 3600),
    "pre_seconds": (0, 120),
    "post_seconds": (0, 120),
    "walk_distance_meters": (0.1, 5),
    "shutter_seconds": (0, 60),
    "min_body_span_meters": (0.2, 1.5),
    "lying_height_meters": (0.15, 1.2),
}


def load() -> Settings:
    settings = Settings()
    if not PATH.exists():
        return settings
    try:
        raw = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    for name, (low, high) in LIMITS.items():
        if name not in raw:
            continue
        try:
            value = float(raw[name])
        except (TypeError, ValueError):
            continue
        setattr(settings, name, min(high, max(low, value)))
    return settings


def save(settings: Settings) -> None:
    PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
