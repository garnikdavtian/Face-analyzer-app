"""
Core face-analysis engine.

Wraps the DeepFace library with:
  • Structured result dataclass instead of raw dicts
  • Robust error handling & logging
  • Multi-face support out of the box
  • Confidence-score extraction for every attribute
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from deepface import DeepFace

from src.config import model_cfg

logger = logging.getLogger(__name__)


# ── Structured result ───────────────────────────────────────────────
@dataclass
class FaceRegion:
    """Bounding box for a detected face."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_dict(cls, d: dict) -> "FaceRegion":
        return cls(
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            w=int(d.get("w", 0)),
            h=int(d.get("h", 0)),
        )

    @property
    def is_valid(self) -> bool:
        return self.w > 0 and self.h > 0


@dataclass
class FaceAnalysisResult:
    """Structured analysis result for a single face."""

    age: int
    dominant_gender: str
    gender_confidence: float
    dominant_race: str
    race_scores: Dict[str, float]
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    region: FaceRegion

    @classmethod
    def from_deepface(cls, raw: dict) -> "FaceAnalysisResult":
        """Parse a raw DeepFace result dict into a typed dataclass."""
        # Gender confidence
        gender_scores: Dict[str, float] = raw.get("gender", {})
        dominant_gender = raw.get("dominant_gender", "Unknown")
        gender_conf = gender_scores.get(dominant_gender, 0.0) if gender_scores else 0.0

        # Race scores
        race_scores: Dict[str, float] = raw.get("race", {})
        dominant_race = raw.get("dominant_race", "Unknown")

        # Emotion scores
        emotion_scores: Dict[str, float] = raw.get("emotion", {})
        dominant_emotion = raw.get("dominant_emotion", "Unknown")

        # Bounding box
        region_dict = raw.get("region") or raw.get("box") or {}
        region = FaceRegion.from_dict(region_dict)

        return cls(
            age=int(round(raw.get("age", 0))),
            dominant_gender=dominant_gender,
            gender_confidence=round(gender_conf, 2),
            dominant_race=dominant_race,
            race_scores={k: round(v, 2) for k, v in race_scores.items()},
            dominant_emotion=dominant_emotion,
            emotion_scores={k: round(v, 2) for k, v in emotion_scores.items()},
            region=region,
        )

    def summary_dict(self) -> Dict[str, object]:
        """Flat dictionary for tabular display."""
        return {
            "Age": self.age,
            "Gender": f"{self.dominant_gender} ({self.gender_confidence:.0f}%)",
            "Race": self.dominant_race,
            "Emotion": self.dominant_emotion,
        }

    def label(self) -> str:
        """Short label for bounding-box overlay."""
        return (
            f"{self.age}y | {self.dominant_gender} | "
            f"{self.dominant_emotion}"
        )


# ── Analysis engine ─────────────────────────────────────────────────
def analyze_faces(
    image: np.ndarray,
    backend: str = model_cfg.default_backend,
    actions: Tuple[str, ...] = model_cfg.actions,
    enforce_detection: bool = model_cfg.enforce_detection,
) -> List[FaceAnalysisResult]:
    """
    Run DeepFace analysis on *image* and return structured results
    for **every** detected face.

    Parameters
    ----------
    image : np.ndarray
        BGR image (as returned by OpenCV).
    backend : str
        Face-detection backend identifier.
    actions : tuple of str
        Analysis actions to perform.
    enforce_detection : bool
        Whether to raise on missing faces.

    Returns
    -------
    list[FaceAnalysisResult]
        One entry per detected face.  Empty list on failure.
    """
    try:
        raw_results = DeepFace.analyze(
            img_path=image,
            actions=actions,
            enforce_detection=enforce_detection,
            detector_backend=backend,
        )
        if not isinstance(raw_results, list):
            raw_results = [raw_results]

        faces = [FaceAnalysisResult.from_deepface(r) for r in raw_results]
        logger.info("Detected %d face(s) using backend '%s'.", len(faces), backend)
        return faces

    except Exception:
        logger.exception("DeepFace analysis failed (backend=%s).", backend)
        return []
