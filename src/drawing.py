"""
OpenCV drawing utilities for annotating video frames.

Provides visually polished bounding boxes with semi-transparent label
backgrounds, configurable via `DrawConfig`.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.analyzer import FaceAnalysisResult
from src.config import draw_cfg


def draw_annotations(
    frame: np.ndarray,
    faces: List[FaceAnalysisResult],
) -> np.ndarray:
    """
    Draw bounding boxes and attribute labels on *frame* (in-place).

    Parameters
    ----------
    frame : np.ndarray
        BGR video frame.
    faces : list[FaceAnalysisResult]
        Analysis results for each detected face.

    Returns
    -------
    np.ndarray
        The annotated frame (same reference as input).
    """
    for face in faces:
        region = face.region
        if not region.is_valid:
            continue

        x, y, w, h = region.x, region.y, region.w, region.h

        # Bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            draw_cfg.box_color,
            draw_cfg.box_thickness,
        )

        # Label background
        label = face.label()
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            draw_cfg.font_scale,
            draw_cfg.font_thickness,
        )
        pad = draw_cfg.label_padding
        label_y = max(y - th - 2 * pad, 0)

        # Semi-transparent background rectangle
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, label_y),
            (x + tw + 2 * pad, label_y + th + 2 * pad),
            draw_cfg.label_bg_color,
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Text
        cv2.putText(
            frame,
            label,
            (x + pad, label_y + th + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            draw_cfg.font_scale,
            draw_cfg.font_color,
            draw_cfg.font_thickness,
            cv2.LINE_AA,
        )

    return frame
