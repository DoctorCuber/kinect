"""Score saved clips with the trained infrared person model."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "person_classifier.pt"
CLIPS_DIR = Path(__file__).resolve().parent / "clips"
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
PERSON_THRESHOLD = 0.6
CLIP_MEAN_THRESHOLD = 0.6
CLIP_FRAME_FRACTION = 0.25


def decide(frames: list[dict[str, float]]) -> dict:
    """Turn per-frame class probabilities into one clip decision.

    Person means a standing human. Unfeasible means the picture looks like a person
    who is not standing, such as someone lying down. That label is used only when
    the model was trained with it and is reasonably sure.
    """
    if not frames:
        return {
            "label": "no_person",
            "confidence": 0.0,
            "presence": 0.0,
            "frames": 0,
            "person_frames": 0,
        }
    names = list(frames[0])
    count = len(frames)
    means = {name: sum(frame.get(name, 0.0) for frame in frames) / count for name in names}
    person_frames = sum(1 for frame in frames if frame.get("person", 0.0) >= PERSON_THRESHOLD)
    person_ok = means.get("person", 0.0) >= CLIP_MEAN_THRESHOLD and person_frames / count >= CLIP_FRAME_FRACTION
    unfeasible_frames = sum(1 for frame in frames if frame.get("unfeasible", 0.0) >= PERSON_THRESHOLD)
    unfeasible_ok = (
        "unfeasible" in means
        and means["unfeasible"] >= CLIP_MEAN_THRESHOLD
        and unfeasible_frames / count >= CLIP_FRAME_FRACTION
        and means["unfeasible"] >= means.get("person", 0.0)
    )
    if unfeasible_ok:
        label = "unfeasible"
        confidence = means["unfeasible"]
    elif person_ok:
        label = "person"
        confidence = means["person"]
    else:
        label = "no_person"
        confidence = means.get("no_person", 1.0 - means.get("person", 0.0))
    if "no_person" in means:
        presence = 1.0 - means["no_person"]
    else:
        presence = means.get("person", 0.0) + means.get("unfeasible", 0.0)
    presence = min(1.0, max(0.0, float(presence)))
    return {
        "label": label,
        "confidence": float(confidence),
        "presence": presence,
        "frames": count,
        "person_frames": person_frames,
    }


def frames_dir_for(clip: Path) -> Path:
    return clip.with_name(clip.stem + "_frames")


def load_clip_frames(clip: Path, limit: int = 24) -> tuple[list[np.ndarray], str]:
    """Raw infrared frames saved beside the clip, or samples from the video."""
    folder = frames_dir_for(clip)
    paths = sorted(folder.glob("*.jpg")) if folder.exists() else []
    if paths:
        if len(paths) > limit:
            indexes = [round(i * (len(paths) - 1) / (limit - 1)) for i in range(limit)]
            paths = [paths[index] for index in indexes]
        images = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        return images, "raw"

    images = []
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        return images, "missing"
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 12)
    step = max(1, int(round(fps / 2)))
    index = 0
    while len(images) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            images.append(frame)
        index += 1
        if frame_count and index >= frame_count:
            break
    capture.release()
    return images, "video"


device_label = "CPU"


def load_model(model_path: Path = MODEL_PATH):
    """Load the trained classifier. Raises FileNotFoundError if it has not been trained."""
    global device_label
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    from compute_device import NpuPersonModel, choose_compute

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=None)
    classifier = model.classifier
    last_linear = max(index for index, layer in enumerate(classifier) if isinstance(layer, torch.nn.Linear))
    in_features = classifier[last_linear].in_features
    classifier[last_linear] = torch.nn.Linear(in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = weights.transforms()
    classes = list(checkpoint["classes"])
    kind, label, train_device = choose_compute()
    if kind == "npu":
        try:
            device_label = label
            return NpuPersonModel(model, transform, classes), transform, classes, "npu"
        except Exception:
            kind = "cpu"
            label = "CPU"
            train_device = "cpu"
    device = torch.device(train_device)
    model.to(device)
    device_label = label
    return model, transform, classes, device


def class_probabilities(model, transform, classes: list[str], device, image_bgr: np.ndarray) -> dict[str, float]:
    if getattr(model, "uses_npu", False):
        return model.probabilities(image_bgr)
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    return {name: float(probabilities[index].item()) for index, name in enumerate(classes)}


def score_clip(clip: Path, model=None, transform=None, classes=None, device=None) -> dict:
    images, source = load_clip_frames(clip)
    if model is None:
        model, transform, classes, device = load_model()
    probabilities = [class_probabilities(model, transform, classes, device, image) for image in images]
    result = decide(probabilities)
    result["image_label"] = result["label"]
    result["image_confidence"] = result["confidence"]
    result["pose"] = "ok"
    pose_path = clip.with_name(clip.stem + "_pose.json")
    if pose_path.exists():
        try:
            counts = json.loads(pose_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts = {}
        from pose_check import dominant_problem

        problem = dominant_problem(counts if isinstance(counts, dict) else {})
        if problem:
            total = sum(int(value) for value in counts.values())
            result["label"] = "unfeasible"
            result["pose"] = problem
            result["confidence"] = int(counts.get(problem, 0)) / total if total else 0.0
    result["source"] = source
    result["clip"] = str(clip)
    clip.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def score_folder(folder: Path = CLIPS_DIR) -> list[dict]:
    clips = sorted(folder.glob("sleepwalk_*.mp4")) + sorted(folder.glob("sleepwalk_*.avi"))
    if not clips:
        return []
    model, transform, classes, device = load_model()
    return [score_clip(clip, model, transform, classes, device) for clip in clips]


def clear_clip_history(folder: Path = CLIPS_DIR, keep: list[Path] | None = None) -> int:
    """Remove saved sleepwalk clips and their scores.

    Training pictures and the model file are left alone. An open recording passed in
    ``keep`` is left alone too, including its score file and raw-frame folder.
    """
    folder = folder.resolve()
    protected = {DATASET_DIR.resolve(), MODEL_PATH.resolve(), MODEL_PATH.parent.resolve()}
    if folder in protected or MODEL_PATH.resolve().is_relative_to(folder) or DATASET_DIR.resolve().is_relative_to(folder):
        raise ValueError("Refusing to clear training data")
    kept_names: set[str] = set()
    for path in keep or []:
        kept_names.add(path.name)
        kept_names.add(path.with_suffix(".json").name)
        kept_names.add(path.with_name(path.stem + "_pose.json").name)
        kept_names.add(path.stem + "_frames")
    if not folder.exists():
        return 0
    removed = 0
    for path in list(folder.iterdir()):
        if path.name in kept_names:
            continue
        clip_file = path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".json"}
        frame_dir = path.is_dir() and path.name.endswith("_frames")
        if path.name.startswith("sleepwalk_") and (clip_file or frame_dir):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
    return removed


def result_text(result: dict) -> str:
    labels = {"person": "Person", "no_person": "No person", "unfeasible": "Unfeasible"}
    label = labels.get(result["label"], result["label"])
    percent = int(round(result["confidence"] * 100))
    name = Path(result["clip"]).name
    note = ""
    if result.get("source") == "video":
        note = "  (clip only, no raw frames)"
    return f"{label}  {percent}%   {name}{note}"
