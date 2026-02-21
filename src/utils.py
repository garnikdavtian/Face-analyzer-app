"""
Shared utility functions for image I/O, format conversion,
and results export.
"""

from __future__ import annotations

import io
from typing import List

import cv2
import numpy as np
import pandas as pd

from src.analyzer import FaceAnalysisResult


def bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes into a BGR OpenCV image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from the provided bytes.")
    return img


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for display in Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def results_to_dataframe(
    faces: List[FaceAnalysisResult],
    source_name: str = "",
) -> pd.DataFrame:
    """
    Convert a list of analysis results into a tidy DataFrame.

    Parameters
    ----------
    faces : list[FaceAnalysisResult]
        Analysis results (possibly from multiple images).
    source_name : str, optional
        Image filename to attach to each row.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for idx, face in enumerate(faces, start=1):
        row = face.summary_dict()
        if source_name:
            row["Source"] = source_name
        row["Face #"] = idx
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
