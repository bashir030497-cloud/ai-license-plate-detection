import streamlit as st
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

model = YOLO("best.pt")

# ---- Fetch TURN credentials from Metered ----
APP_NAME = "ai-license-plate.metered.live"      # <-- yahan apna app name daalo
API_KEY = "msbtQg2pGWziMrT3dE1k2nsZzDJgLj3rvYQGOO0I4slmpZpe"        # <-- yahan apni API key daalo

@st.cache_resource
def get_ice_servers():
    url = f"https://{APP_NAME}.metered.live/api/v1/turn/credentials?apiKey={API_KEY}"
    response = requests.get(url)
    ice_servers = response.json()
    return ice_servers

ice_servers = get_ice_servers()

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": ice_servers,
    "iceTransportPolicy": "relay",
})

class PlateDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

st.title("AI License Plate Detection")

webrtc_streamer(
    key="plate-detection",
    video_processor_factory=PlateDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
