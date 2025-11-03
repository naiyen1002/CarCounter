import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import yt_dlp
import os

st.set_page_config(page_title="Car Counter", layout="centered")
st.title("Vehicle Detection & Counting")    

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar settings
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.2, 0.9, 0.5, 0.05)

# Choose video input
video_path = None
url = st.text_input("Enter YouTube link:")
if url:
    st.info("Downloading video from YouTube...")
    try:
        video_path = os.path.join(tempfile.gettempdir(), "video.mp4")
        ydl_opts = {"format": "bestvideo[ext=mp4]", "outtmpl": video_path, "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        st.success("Download complete ✅")
    except Exception as e:
        st.error(f"Download failed: {e}")

# Run detection
if video_path:
    stframe = st.empty()
    cap = cv2.VideoCapture(video_path)
    unique_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml")

        # Draw results
        for r in results:
            boxes = r.boxes
            if boxes.id is None:
                continue

            for box, id_, cls_, conf_ in zip(boxes.xyxy, boxes.id, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                id_ = int(id_)
                cls_ = int(cls_)
                conf_ = float(conf_)

                label = f"{model.names[cls_]} {int(conf_ * 100)}%"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                unique_ids.add(id_)

        # Display count
        cv2.putText(frame, f"Total Vehicle: {len(unique_ids)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.success(f"Total vehicles detected: {len(unique_ids)}")

else:
    st.info("Upload a YouTube link to start detection.")
