<div align="center">

# Face Analyzer

**Multi-Attribute Facial Analysis App — Age · Gender · Race · Emotion**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.89+-4A90D9?logo=tensorflow&logoColor=white)](https://github.com/serengil/deepface)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![CI](https://github.com/garnikdavtian/Face-analyzer-app/actions/workflows/ci.yml/badge.svg)](https://github.com/garnikdavtian/Face-analyzer-app/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<br>

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Docker](#-docker) · [Demo](#-demo)

</div>

---

## Overview

**Face Analyzer** is an interactive computer-vision application that performs **multi-attribute facial analysis** from uploaded images, webcam snapshots, or live video streams. Built on top of [DeepFace](https://github.com/serengil/deepface)'s ensemble of state-of-the-art models, it detects and classifies:

| Attribute | Model Backbone | Output |
|-----------|---------------|--------|
| **Age** | VGG-Face / DeepFace ensemble | Estimated age (years) |
| **Gender** | VGG-Face | Male / Female + confidence % |
| **Race** | VGG-Face | 6 categories + score distribution |
| **Emotion** | Facial Expression Recognition | 7 categories + score distribution |

The app goes beyond simple prediction labels by exposing **full confidence distributions** for race and emotion, enabling users to understand the model's certainty profile.

---

## ✨ Features

- **Three analysis modes**
  - 📁 **Batch Upload** — drag & drop multiple images with aggregate CSV export
  - 📷 **Webcam Capture** — one-click snapshot analysis
  - 🎥 **Live Camera** — real-time annotated video stream with FPS counter

- **Multi-face detection** — handles group photos; all detected faces are analyzed independently

- **Confidence transparency** — expandable bar charts show the full probability distribution for every attribute

- **Configurable backends** — switch between `opencv`, `retinaface`, `mtcnn`, `ssd`, and `mediapipe` detection backends at runtime

- **Polished UI** — dark-mode gradient theme, metric cards, animated hover effects, and responsive layout

- **Production-ready**
  - Dockerized deployment with health checks
  - CI pipeline (GitHub Actions) with linting + tests
  - Pinned dependencies for reproducible builds
  - Structured logging throughout

---

## 🏗️ Architecture

```
Face-analyzer-app/
│
├── app.py                    # Streamlit entry point (thin orchestrator)
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized dataclass-based configuration
│   ├── analyzer.py           # Core analysis engine + structured result types
│   ├── drawing.py            # OpenCV annotation utilities (bounding boxes)
│   └── utils.py              # Image I/O, format conversion, CSV export
│
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py      # Unit tests (pytest)
│
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── Dockerfile                # Multi-stage container build
├── requirements.txt          # Pinned Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # ← You are here
```


## 🚀 Quick Start

### Prerequisites

- Python **3.10+**
- Webcam (for capture / live modes)

### 1. Clone & Install

```bash
git clone https://github.com/garnikdavtian/Face-analyzer-app.git
cd Face-analyzer-app

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

The app opens at **`http://localhost:8501`**.

### 3. Test

```bash
pytest tests/ -v
```

---

## 🐳 Docker

```bash
# Build
docker build -t face-analyzer .

# Run
docker run -p 8501:8501 face-analyzer
```

> **Note:** Webcam / live-camera modes require host camera pass-through, which varies by OS and Docker runtime.

---

## 🎬 Demo

[Watch the demo video](https://youtu.be/xDThCqsSK-Q)

---

## 🔧 Configuration

All configurable parameters live in [`src/config.py`](src/config.py):

| Config Class | Key Parameters | Default |
|-------------|---------------|---------|
| `CameraConfig` | `frame_width`, `frame_height`, `analysis_interval` | 640×480, every 15 frames |
| `ModelConfig` | `default_backend`, `available_backends`, `enforce_detection` | `opencv`, 5 backends, `False` |
| `UIConfig` | `page_title`, `thumbnail_width`, `upload_extensions` | 280px, jpg/png/webp |
| `DrawConfig` | `box_color`, `font_scale`, `label_padding` | green-500, 0.55, 4px |

---

## 🧪 Testing

The test suite covers:

- **`FaceRegion`** — bounding-box parsing and validation
- **`FaceAnalysisResult`** — raw DeepFace output parsing, confidence extraction, label formatting
- **Utilities** — BGR↔RGB conversion, invalid image handling, DataFrame construction

```bash
pytest tests/ -v --tb=short
```

---

## 🛣️ Roadmap

- [ ] **Face comparison** — verify whether two images contain the same person
- [ ] **Model benchmarking dashboard** — compare detection backends on accuracy/speed
- [ ] **GPU acceleration** — CUDA-enabled inference for > 30 FPS live analysis
- [ ] **REST API** — FastAPI wrapper for programmatic access
- [ ] **Streamlit Cloud deployment** — one-click deploy configuration

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

