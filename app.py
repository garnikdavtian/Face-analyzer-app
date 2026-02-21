"""
Face Analyzer — Multi-Attribute Facial Analysis App
====================================================

A Streamlit application that performs real-time and static-image
face analysis using DeepFace.  Detects **age**, **gender**, **race**,
and **emotion** with confidence scores, supports multiple faces per
image, and provides CSV export.

Modes
-----
1. **Upload Images** — batch analysis of one or more images
2. **Webcam Capture** — single-shot capture and analysis
3. **Live Camera**   — real-time video stream with annotated bounding boxes
"""

from __future__ import annotations

import logging
import time
from typing import List

import cv2
import pandas as pd
import streamlit as st

from src.analyzer import FaceAnalysisResult, analyze_faces
from src.config import camera_cfg, model_cfg, ui_cfg
from src.drawing import draw_annotations
from src.utils import (
    bgr_to_rgb,
    bytes_to_bgr,
    dataframe_to_csv_bytes,
    results_to_dataframe,
)

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title=ui_cfg.page_title,
    page_icon=ui_cfg.page_icon,
    layout=ui_cfg.layout,
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #27293d 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-label {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: #374151;
        margin-top: 8px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-success {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
    }
    .status-info {
        background: rgba(59,130,246,0.15);
        color: #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ──────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔬 Face Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Multi-attribute facial analysis · Age · Gender · Race · Emotion · Confidence Scores"
    "</p>",
    unsafe_allow_html=True,
)


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {ui_cfg.sidebar_title}")
    st.markdown("---")

    mode = st.radio(
        "**Analysis Mode**",
        ["📁 Upload Images", "📷 Webcam Capture", "🎥 Live Camera"],
        index=0,
    )
    backend = st.selectbox(
        "**Detection Backend**",
        model_cfg.available_backends,
        index=0,
        help="RetinaFace is the most accurate; OpenCV is the fastest.",
    )
    mirror = st.checkbox("Mirror Camera", value=camera_cfg.mirror)

    st.markdown("---")
    st.markdown(
        """
        <div style='color:#9ca3af; font-size:0.8rem;'>
        <b>Powered by</b><br>
        DeepFace · OpenCV · Streamlit<br><br>
        <b>Author</b>: Garnik Davtian<br>
        <b>Version</b>: 2.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Helper: render face results ─────────────────────────────────────
def render_face_cards(faces: List[FaceAnalysisResult]) -> None:
    """Display analysis results as styled metric cards with confidence bars."""
    for idx, face in enumerate(faces, start=1):
        if len(faces) > 1:
            st.markdown(f"#### Face #{idx}")

        cols = st.columns(4)

        # Age
        with cols[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Age</div>
                    <div class="metric-value">{face.age}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Gender
        with cols[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Gender</div>
                    <div class="metric-value">{face.dominant_gender}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{face.gender_confidence}%"></div>
                    </div>
                    <div class="metric-label" style="margin-top:4px">{face.gender_confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Race
        with cols[2]:
            top_race_conf = face.race_scores.get(face.dominant_race, 0.0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Race</div>
                    <div class="metric-value">{face.dominant_race}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{top_race_conf}%"></div>
                    </div>
                    <div class="metric-label" style="margin-top:4px">{top_race_conf:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Emotion
        with cols[3]:
            top_emo_conf = face.emotion_scores.get(face.dominant_emotion, 0.0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Emotion</div>
                    <div class="metric-value">{face.dominant_emotion}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{top_emo_conf}%"></div>
                    </div>
                    <div class="metric-label" style="margin-top:4px">{top_emo_conf:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Expandable confidence distributions
        with st.expander("📊 View full confidence distributions"):
            ecol, rcol = st.columns(2)
            with ecol:
                st.markdown("**Emotion Scores**")
                emo_df = pd.DataFrame(
                    list(face.emotion_scores.items()),
                    columns=["Emotion", "Score (%)"],
                ).sort_values("Score (%)", ascending=False)
                st.bar_chart(emo_df.set_index("Emotion"))
            with rcol:
                st.markdown("**Race Scores**")
                race_df = pd.DataFrame(
                    list(face.race_scores.items()),
                    columns=["Race", "Score (%)"],
                ).sort_values("Score (%)", ascending=False)
                st.bar_chart(race_df.set_index("Race"))


# ══════════════════════════════════════════════════════════════════════
#  MODE: Upload Images
# ══════════════════════════════════════════════════════════════════════
if mode == "📁 Upload Images":
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📁 Batch Image Analysis")

    uploaded_files = st.file_uploader(
        "Drop images here",
        type=ui_cfg.upload_extensions,
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_faces: List[FaceAnalysisResult] = []
        all_rows: list = []
        progress = st.progress(0, text="Analyzing images…")

        for i, uploaded_file in enumerate(uploaded_files):
            img = bytes_to_bgr(uploaded_file.read())
            faces = analyze_faces(img, backend=backend)

            st.markdown(f"---\n#### 🖼️ {uploaded_file.name}")
            col_img, col_res = st.columns([1, 2])

            with col_img:
                st.image(bgr_to_rgb(img), width=ui_cfg.thumbnail_width)

            with col_res:
                if faces:
                    st.markdown(
                        f'<span class="status-badge status-success">'
                        f"{len(faces)} face(s) detected</span>",
                        unsafe_allow_html=True,
                    )
                    render_face_cards(faces)
                    for face in faces:
                        row = face.summary_dict()
                        row["Source"] = uploaded_file.name
                        all_rows.append(row)
                else:
                    st.warning("No faces detected in this image.")

            progress.progress((i + 1) / len(uploaded_files))

        # Aggregate results table
        if all_rows:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 📋 Aggregate Results")
            df = pd.DataFrame(all_rows)
            st.dataframe(df, use_container_width=True)

            csv = dataframe_to_csv_bytes(df)
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name="face_analysis_results.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════
#  MODE: Webcam Capture
# ══════════════════════════════════════════════════════════════════════
elif mode == "📷 Webcam Capture":
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📷 Webcam Capture")
    st.info("Click the camera icon below to take a snapshot.")

    picture = st.camera_input("Capture Photo")

    if picture:
        img = bytes_to_bgr(picture.getvalue())
        if mirror:
            img = cv2.flip(img, 1)

        with st.spinner("Analyzing face(s)…"):
            faces = analyze_faces(img, backend=backend)

        st.image(bgr_to_rgb(img), caption="Captured Photo", width=400)

        if faces:
            st.markdown(
                f'<span class="status-badge status-success">'
                f"{len(faces)} face(s) detected</span>",
                unsafe_allow_html=True,
            )
            render_face_cards(faces)
        else:
            st.warning("No faces detected. Try a different angle or backend.")


# ══════════════════════════════════════════════════════════════════════
#  MODE: Live Camera
# ══════════════════════════════════════════════════════════════════════
elif mode == "🎥 Live Camera":
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🎥 Real-Time Face Analysis")

    col_ctrl, col_info = st.columns([1, 3])
    with col_ctrl:
        run = st.checkbox("▶️ Start Camera", key="live_cam")
    with col_info:
        st.markdown(
            f'<span class="status-badge status-info">Backend: {backend}</span>',
            unsafe_allow_html=True,
        )

    FRAME = st.empty()
    STATS = st.empty()

    cap = cv2.VideoCapture(camera_cfg.device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.frame_height)

    last_faces: List[FaceAnalysisResult] = []
    frame_count = 0
    fps_start = time.time()

    try:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access camera.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_count += 1

            # Analyze periodically
            if frame_count % camera_cfg.analysis_interval == 0:
                faces = analyze_faces(frame, backend=backend)
                if faces:
                    last_faces = faces

            # Draw annotations
            draw_annotations(frame, last_faces)

            # FPS calculation
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            FRAME.image(bgr_to_rgb(frame))
            STATS.markdown(
                f"**Faces:** {len(last_faces)} &nbsp;|&nbsp; "
                f"**FPS:** {fps:.1f} &nbsp;|&nbsp; "
                f"**Backend:** {backend}"
            )
    finally:
        cap.release()
        logger.info("Camera released.")
