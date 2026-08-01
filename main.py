import streamlit as st
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from twilio.rest import Client

# ============================
# Streamlit Page Configuration
# ============================

st.set_page_config(
    page_title="AI License Plate Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI License Plate Detection")
st.write("Real-time License Plate Detection using YOLOv8")

# ============================
# Load YOLO Model
# ============================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ============================
# Twilio Credentials
# (Stored in Streamlit Secrets)
# ============================

ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]

# ============================
# ICE Servers
# ============================

@st.cache_resource
def get_ice_servers():
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    token = client.tokens.create()
    return token.ice_servers

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    }
)

# ============================
# Video Processor
# ============================

class PlateDetector(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = model(image)

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# ============================
# Webcam
# ============================

webrtc_streamer(
    key="license-plate",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=PlateDetector,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
