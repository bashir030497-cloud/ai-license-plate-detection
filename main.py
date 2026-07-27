import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from twilio.rest import Client
import av

model = YOLO("best.pt")

# ---- Twilio credentials ----
ACCOUNT_SID = st.secrets["ACCOUNT_SID"]
AUTH_TOKEN = st.secrets["AUTH_TOKEN"]

@st.cache_resource
def get_ice_servers():
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    token = client.tokens.create()
    return token.ice_servers

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
