import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

model = YOLO("best.pt")

class PlateDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)
        annotated = results[0].plot()  # draws boxes automatically

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

st.title("AI License Plate Detection")

webrtc_streamer(
    key="plate-detection",
    video_processor_factory=PlateDetector,
    media_stream_constraints={"video": True, "audio": False},
)
