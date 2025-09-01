import streamlit as st
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace

# Page config
st.set_page_config(page_title="Age, Gender, Race, Emotion Detector", layout="wide")

st.title("Age, Gender, Race, and Emotion Detection")
st.write("Upload images or use your webcam to analyze Age, Gender, Race, and Emotion.")

# Sidebar Info
st.sidebar.markdown("## Face Analyzer")
st.sidebar.markdown("""
This application uses **DeepFace** for analyzing:
- Age
- Gender
- Race
- Emotion  

Available modes:
- **Upload Images** — upload one or multiple images  
- **Webcam Capture** — take a photo with your webcam  
- **Live Camera** — real-time detection from webcam feed  
""")

# Cached DeepFace Model
@st.cache_resource
def get_deepface_model():
    return DeepFace.build_model("VGG-Face")

model = get_deepface_model()

# Analyzer Function
def analyze_image(img, backend="opencv"):
    """Analyze image using DeepFace and return list of results."""
    try:
        results = DeepFace.analyze(
            img,
            actions=("age", "gender", "race", "emotion"),
            enforce_detection=False,
            detector_backend=backend
        )
        if not isinstance(results, list):
            results = [results]
        return results
    except Exception as e:
        st.warning(f"DeepFace error: {e}")
        return None

#Sidebar Settings
mode = st.sidebar.radio("Mode", ["Upload Images", "Webcam Capture", "Live Camera"])
backend = st.sidebar.selectbox("Face Detector Backend", ["opencv", "retinaface", "mtcnn"])
mirror = st.sidebar.checkbox("Mirror Camera", value=True)

# Upload Images Mode
if mode == "Upload Images":
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        data = {"Name": [], "Age": [], "Gender": [], "Race": [], "Emotion": []}
        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            file_bytes = uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = analyze_image(img, backend=backend)
            if results:
                result = results[0]
                data["Name"].append(uploaded_file.name)
                data["Age"].append(round(result.get("age", 0)))
                data["Gender"].append(result.get("dominant_gender"))
                data["Race"].append(result.get("dominant_race"))
                data["Emotion"].append(result.get("dominant_emotion"))
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=uploaded_file.name, width=250)
            else:
                st.error(f"Error analyzing {uploaded_file.name}")

            progress.progress((i + 1) / len(uploaded_files))

        # Display results in table
        df = pd.DataFrame(data)
        st.subheader("Analysis Results")
        st.dataframe(df)

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", data=csv, file_name="analysis_results.csv", mime="text/csv")

# Webcam Capture Mode
elif mode == "Webcam Capture":
    st.write("Capture a photo with your webcam.")
    picture = st.camera_input("Capture Photo")
    if picture:
        file_bytes = picture.getvalue()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if mirror:
            img = cv2.flip(img, 1)  # flip horizontal

        results = analyze_image(img, backend=backend)
        if results:
            result = results[0]
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured Photo", width=300)
            df = pd.DataFrame([{
                "Age": round(result.get("age", 0)),
                "Gender": result.get("dominant_gender"),
                "Race": result.get("dominant_race"),
                "Emotion": result.get("dominant_emotion")
            }])
            st.subheader("Analysis Result")
            st.dataframe(df)
        else:
            st.error("Error analyzing captured photo")

# Live Camera Mode
elif mode == "Live Camera":
    st.write("Live age, gender, race, and emotion detection")
    run = st.checkbox("Start Camera")
    FRAME = st.empty()
    INFO = st.empty()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_results = []
    frame_count = 0

    try:
        while run:
            ret, frame = cap.read()
            if mirror:
                frame = cv2.flip(frame, 1)  # flip horizontal

            if not ret:
                st.error("Failed to access camera")
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Analize only every 30th frame
                results = analyze_image(frame, backend=backend)
                if results:
                    last_results = results

            # Draw rectangles and labels
            for face in last_results:
                region = face.get("region") or face.get("box") or {}
                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                w = int(region.get("w", 0))
                h = int(region.get("h", 0))
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{face.get('age','?')} | {face.get('dominant_gender','?')} | {face.get('dominant_race','?')} | {face.get('dominant_emotion','?')}"
                    cv2.putText(frame, label, (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            INFO.text(f"Faces detected: {len(last_results)} | Backend: {backend}")
            FRAME.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
