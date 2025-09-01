# Face Analyzer App

## Overview
This project is an interactive web application built with **Streamlit** that performs real-time and static image analysis using the **DeepFace** library. It can detect and display information about **age, gender, race, and emotion** from uploaded images, webcam captures, or live camera feed.  

The project was developed as a portfolio piece to showcase computer vision and machine learning integration with a user-friendly interface.

---

## Features
- **Multiple modes of operation**:
  - **Upload Images**: Analyze one or multiple uploaded images.
  - **Webcam Capture**: Take a snapshot with your webcam and analyze it.
  - **Live Camera**: Real-time detection with bounding boxes and labels.

- **Customizable settings**:
  - Choose between different face detection backends (`opencv`, `retinaface`, `mtcnn`).
  - Optional camera mirroring for a natural experience.

- **Visualization**:
  - Bounding boxes with labels (age, gender, race, emotion) drawn on live video.
  - Tabular display of analysis results.

- **Export**:
  - Download results as a CSV file when analyzing multiple uploaded images.

---

## Demo
A demonstration video showcasing the main features of the application is available here:  

[![Watch the demo](https://img.youtube.com/vi/mC8AMd8nStA/0.jpg)](https://www.youtube.com/watch?v=mC8AMd8nStA)

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/face-analyzer-app.git
   cd face-analyzer-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open the provided local URL in your browser.

3. Choose a mode from the sidebar:
   - Upload images
   - Capture from webcam
   - Live camera stream


---

## Notes
- Some backends may require additional dependencies (e.g., `mtcnn`, `retinaface`).
- For deployment on Streamlit Cloud or other environments, ensure camera access is supported.
- The project was created for portfolio purposes and demonstrates integration of face analysis into a user-facing application.

---
