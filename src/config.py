"""
Centralized configuration for the Face Analyzer application.

All tuneable hyper-parameters, UI defaults, and model settings are
consolidated here so that they can be modified without touching
application logic.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class CameraConfig:
    """Settings for the live-camera / webcam capture pipeline."""

    frame_width: int = 640
    frame_height: int = 480
    analysis_interval: int = 15          # analyse every N-th frame
    device_index: int = 0                # default camera device
    mirror: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """DeepFace analysis settings."""

    actions: Tuple[str, ...] = ("age", "gender", "race", "emotion")
    enforce_detection: bool = False
    default_backend: str = "opencv"
    available_backends: Tuple[str, ...] = (
        "opencv",
        "retinaface",
        "mtcnn",
        "ssd",
        "mediapipe",
    )


@dataclass(frozen=True)
class UIConfig:
    """Streamlit UI constants."""

    page_title: str = "Face Analyzer — Multi-Attribute Facial Analysis"
    page_icon: str = "🔬"
    layout: str = "wide"
    sidebar_title: str = "⚙️ Settings"
    upload_extensions: List[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"]
    )
    thumbnail_width: int = 280


@dataclass(frozen=True)
class DrawConfig:
    """Bounding-box / label drawing parameters."""

    box_color: Tuple[int, int, int] = (34, 197, 94)       # green-500
    box_thickness: int = 2
    font_scale: float = 0.55
    font_color: Tuple[int, int, int] = (255, 255, 255)
    font_thickness: int = 1
    label_bg_color: Tuple[int, int, int] = (34, 197, 94)  # green-500
    label_padding: int = 4


# ── singleton instances ─────────────────────────────────────────────
camera_cfg = CameraConfig()
model_cfg = ModelConfig()
ui_cfg = UIConfig()
draw_cfg = DrawConfig()
