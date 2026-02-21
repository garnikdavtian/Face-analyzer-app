"""
Unit tests for the core analysis engine and utilities.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analyzer import FaceAnalysisResult, FaceRegion
from src.utils import bgr_to_rgb, bytes_to_bgr, results_to_dataframe


# ── FaceRegion ──────────────────────────────────────────────────────
class TestFaceRegion:
    def test_from_dict_valid(self):
        region = FaceRegion.from_dict({"x": 10, "y": 20, "w": 100, "h": 120})
        assert region.x == 10
        assert region.y == 20
        assert region.w == 100
        assert region.h == 120
        assert region.is_valid

    def test_from_dict_empty(self):
        region = FaceRegion.from_dict({})
        assert region.x == 0
        assert not region.is_valid

    def test_from_dict_zero_size(self):
        region = FaceRegion.from_dict({"x": 5, "y": 5, "w": 0, "h": 0})
        assert not region.is_valid


# ── FaceAnalysisResult ──────────────────────────────────────────────
class TestFaceAnalysisResult:
    @pytest.fixture
    def raw_deepface_output(self) -> dict:
        return {
            "age": 28.5,
            "gender": {"Woman": 2.3, "Man": 97.7},
            "dominant_gender": "Man",
            "race": {
                "asian": 1.0, "indian": 2.0, "black": 0.5,
                "white": 90.0, "middle eastern": 4.0, "latino hispanic": 2.5,
            },
            "dominant_race": "white",
            "emotion": {
                "angry": 0.1, "disgust": 0.0, "fear": 0.2,
                "happy": 95.0, "sad": 1.0, "surprise": 0.5, "neutral": 3.2,
            },
            "dominant_emotion": "happy",
            "region": {"x": 50, "y": 60, "w": 200, "h": 220},
        }

    def test_from_deepface(self, raw_deepface_output):
        result = FaceAnalysisResult.from_deepface(raw_deepface_output)
        assert result.age == 28  # rounded from 28.5
        assert result.dominant_gender == "Man"
        assert result.gender_confidence == 97.7
        assert result.dominant_race == "white"
        assert result.dominant_emotion == "happy"
        assert result.region.is_valid

    def test_summary_dict_keys(self, raw_deepface_output):
        result = FaceAnalysisResult.from_deepface(raw_deepface_output)
        summary = result.summary_dict()
        assert set(summary.keys()) == {"Age", "Gender", "Race", "Emotion"}

    def test_label_format(self, raw_deepface_output):
        result = FaceAnalysisResult.from_deepface(raw_deepface_output)
        label = result.label()
        assert "28" in label
        assert "Man" in label
        assert "happy" in label


# ── Utilities ───────────────────────────────────────────────────────
class TestUtils:
    def test_bgr_to_rgb(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # blue channel
        rgb = bgr_to_rgb(bgr)
        assert rgb[0, 0, 2] == 255  # blue moved to channel 2

    def test_bytes_to_bgr_invalid_raises(self):
        with pytest.raises(ValueError, match="Could not decode"):
            bytes_to_bgr(b"not-an-image")

    def test_results_to_dataframe(self):
        face = FaceAnalysisResult(
            age=30,
            dominant_gender="Woman",
            gender_confidence=95.0,
            dominant_race="asian",
            race_scores={"asian": 95.0},
            dominant_emotion="neutral",
            emotion_scores={"neutral": 80.0},
            region=FaceRegion(0, 0, 100, 100),
        )
        df = results_to_dataframe([face], source_name="test.jpg")
        assert len(df) == 1
        assert "Source" in df.columns
        assert df.iloc[0]["Source"] == "test.jpg"
