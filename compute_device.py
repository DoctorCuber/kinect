"""Pick where the person model runs. Inference and training stay on this computer.

Order is NVIDIA GPU, then an NPU, then the CPU. Training needs gradients, and the
NPU path cannot train this model, so a missing NVIDIA GPU makes training use the CPU.
Clip checking still uses the NPU in that case.
"""

from __future__ import annotations

import numpy as np


class NpuPersonModel:
    """MobileNet classifier compiled for an OpenVINO NPU."""

    uses_npu = True

    def __init__(self, torch_model, transform, classes: list[str]) -> None:
        import openvino as ov
        import torch

        torch_model = torch_model.cpu().eval()
        example = torch.zeros(1, 3, 224, 224)
        converted = ov.convert_model(torch_model, example_input=example, dynamo=True)
        core = ov.Core()
        self.compiled = core.compile_model(converted, "NPU")
        self.transform = transform
        self.classes = list(classes)

    def probabilities(self, image_bgr: np.ndarray) -> dict[str, float]:
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch = self.transform(Image.fromarray(rgb)).unsqueeze(0).numpy()
        result = self.compiled(batch)
        logits = np.asarray(next(iter(result.values())), dtype=np.float32).reshape(-1)
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        probabilities = exp / exp.sum()
        return {name: float(probabilities[index]) for index, name in enumerate(self.classes)}

    def probability(self, image_bgr: np.ndarray) -> float:
        return self.probabilities(image_bgr)["person"]


def choose_compute() -> tuple[str, str, str]:
    """Return kind, a label for the window, and the torch device used for training.

    kind is ``cuda``, ``npu``, or ``cpu``. The training device is ``cuda`` or ``cpu``.
    """
    import torch

    if torch.cuda.is_available():
        index = _nvidia_index()
        if index is not None:
            name = torch.cuda.get_device_name(index)
            return "cuda", f"NVIDIA GPU ({name})", f"cuda:{index}"

    npu_name = _npu_name()
    if npu_name:
        return "npu", f"NPU ({npu_name})", "cpu"
    return "cpu", "CPU", "cpu"


def _nvidia_index() -> int | None:
    import torch

    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index).lower()
        if "nvidia" in name or "geforce" in name or "rtx" in name or "quadro" in name:
            return index
    return None


def _npu_name() -> str | None:
    try:
        import openvino as ov
    except Exception:
        return None
    try:
        core = ov.Core()
        devices = [str(device) for device in core.available_devices]
    except Exception:
        return None
    npu_devices = [device for device in devices if device == "NPU" or device.startswith("NPU.")]
    if not npu_devices:
        return None
    try:
        return str(core.get_property(npu_devices[0], "FULL_DEVICE_NAME"))
    except Exception:
        return "Intel AI Boost"
