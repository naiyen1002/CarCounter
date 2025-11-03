# Car Counter

## Overview
A Streamlit web app that detects and counts vehicles in a video using the YOLOv8 model.  
Users can provide a YouTube link, and the app downloads and processes the video in real time, showing detected vehicles with bounding boxes, confidence levels, and total counts.

---

## Approach
- Used **YOLOv8n** for fast and accurate vehicle detection.  
- Applied **ByteTrack** to track unique vehicles and avoid double counting.  
- Implemented the interface with **Streamlit** for easy interaction.  
- Used **OpenCV** for visualization and **yt-dlp** for video downloading.

---

## Tools Used
- **Python**
- **Streamlit**
- **Ultralytics YOLOv8**
- **OpenCV**
- **yt-dlp**

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Enter a YouTube link and start detection.

---

## Requirements
```
streamlit==1.39.0
ultralytics==8.3.25
opencv-python==4.10.0.84
yt-dlp==2024.10.7
numpy==1.26.4
```
