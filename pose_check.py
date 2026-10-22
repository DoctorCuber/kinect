"""Decide whether a Kinect skeleton is a plausible upright person."""

from __future__ import annotations

# Both joints must be tracked. Lengths are meters, loose human ranges.
_BONES = (
    (20, 3, 0.08, 0.45),
    (0, 20, 0.2, 0.8),
    (4, 5, 0.12, 0.55),
    (8, 9, 0.12, 0.55),
    (5, 6, 0.1, 0.5),
    (9, 10, 0.1, 0.5),
    (12, 13, 0.2, 0.75),
    (16, 17, 0.2, 0.75),
    (13, 14, 0.18, 0.7),
    (17, 18, 0.18, 0.7),
)

POSE_NOTES = {
    "cloud": "The joints were bunched into a short cloud, too small to be a person.",
    "lying": "The body was flat, like someone lying in bed, rather than standing up.",
    "proportions": "The limb lengths were not a possible person.",
}


def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def classify_body(body, min_span: float, lying_height: float) -> str:
    """Return ok, cloud, lying, or proportions."""
    if not body.camera_xyz:
        return "ok"
    points = []
    for index, point in enumerate(body.camera_xyz):
        if body.states[index] <= 0:
            continue
        x, y, z = point
        if z <= 0.2:
            continue
        points.append((index, (x, y, z)))
    if len(points) < 4:
        return "ok"
    xs = [point[0] for _index, point in points]
    ys = [point[1] for _index, point in points]
    zs = [point[2] for _index, point in points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    span = (dx * dx + dy * dy + dz * dz) ** 0.5
    horizontal = (dx * dx + dz * dz) ** 0.5
    if span < min_span or (dy > 0.5 and horizontal < 0.18):
        return "cloud"
    by_index = {index: point for index, point in points}
    for start, end, low, high in _BONES:
        if body.states[start] != 2 or body.states[end] != 2:
            continue
        if start not in by_index or end not in by_index:
            continue
        length = _distance(by_index[start], by_index[end])
        if length < low or length > high:
            return "proportions"
    if dy < lying_height and horizontal > max(0.7, dy * 1.5):
        return "lying"
    return "ok"


def dominant_problem(counts: dict) -> str | None:
    """The pose problem that covers at least half of the samples, if any."""
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return None
    kind = max(counts, key=lambda name: int(counts[name]))
    if kind == "ok" or int(counts[kind]) / total < 0.5:
        return None
    return kind
