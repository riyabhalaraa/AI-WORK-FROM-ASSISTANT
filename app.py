import time
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
DISTRACTION_THRESHOLD = 6.2
POSTURE_THRESHOLD_PX = 105
ALERT_DELAY = 5  # seconds

st.set_page_config(page_title="Focus + Posture Monitor", layout="wide")
st.title("Focus + Posture Monitor")
st.write("Detects eye distraction and bad posture from your webcam.")

# Sidebar controls
st.sidebar.header("Settings")
distraction_threshold = st.sidebar.slider(
    "Eye distraction threshold", 1.0, 15.0, float(DISTRACTION_THRESHOLD), 0.1
)
posture_threshold = st.sidebar.slider(
    "Posture threshold (px)", 20, 250, int(POSTURE_THRESHOLD_PX), 1
)
alert_delay = st.sidebar.slider(
    "Alert delay (seconds)", 1, 15, int(ALERT_DELAY), 1
)
show_landmarks = st.sidebar.checkbox("Show landmarks", value=True)


class MonitorProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils
        self.pose_module = mp.solutions.pose

        self.eye_distraction_time = None
        self.posture_distraction_time = None
        self.eye_alert = False
        self.posture_alert = False

    def horizontal_diff(self, p1, p2, frame_width):
        return abs(p1[0] - p2[0]) * frame_width

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Default statuses
        eye_status = "Focused"
        eye_color = (0, 255, 0)

        posture_status = "Good Posture"
        posture_color = (0, 255, 0)
        posture_offset = 0

        # -----------------------------
        # FACE / EYE DISTRACTION
        # -----------------------------
        face_results = self.face_mesh.process(rgb)

        if face_results.multi_face_landmarks:
            facial_landmarks = face_results.multi_face_landmarks[0]

            xaxis_left = int(sum(facial_landmarks.landmark[i].x for i in [469, 470, 471, 472]) / 4 * w)
            yaxis_left = int(sum(facial_landmarks.landmark[i].y for i in [469, 470, 471, 472]) / 4 * h)

            xaxis_right = int(sum(facial_landmarks.landmark[i].x for i in [474, 475, 476, 477]) / 4 * w)
            yaxis_right = int(sum(facial_landmarks.landmark[i].y for i in [474, 475, 476, 477]) / 4 * h)

            top_right = int(facial_landmarks.landmark[475].y * h)
            top_left = int(facial_landmarks.landmark[470].y * h)
            bottom_right = int(facial_landmarks.landmark[477].y * h)
            bottom_left = int(facial_landmarks.landmark[472].y * h)

            center_right = yaxis_right
            center_left = yaxis_left

            right_1 = center_right - top_right
            left_1 = center_left - top_left
            right_2 = bottom_right - center_right
            left_2 = bottom_left - center_left

            distracted = (
                right_1 < distraction_threshold
                or right_2 < distraction_threshold
                or left_1 < distraction_threshold
                or left_2 < distraction_threshold
            )

            if distracted:
                eye_status = "Distracted"
                eye_color = (0, 0, 255)

                if self.eye_distraction_time is None:
                    self.eye_distraction_time = time.time()
                elif time.time() - self.eye_distraction_time > alert_delay and not self.eye_alert:
                    self.eye_alert = True
            else:
                self.eye_distraction_time = None
                self.eye_alert = False
        else:
            eye_status = "Face Not Found"
            eye_color = (0, 165, 255)
            self.eye_distraction_time = None
            self.eye_alert = False

        # -----------------------------
        # POSTURE DETECTION
        # -----------------------------
        pose_results = self.pose.process(rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            left_ear = [
                landmarks[self.pose_module.PoseLandmark.LEFT_EAR.value].x,
                landmarks[self.pose_module.PoseLandmark.LEFT_EAR.value].y
            ]
            left_shoulder = [
                landmarks[self.pose_module.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[self.pose_module.PoseLandmark.LEFT_SHOULDER.value].y
            ]

            posture_offset = self.horizontal_diff(left_ear, left_shoulder, w)

            if posture_offset > posture_threshold:
                posture_status = "Slouching"
                posture_color = (0, 0, 255)

                if self.posture_distraction_time is None:
                    self.posture_distraction_time = time.time()
                elif time.time() - self.posture_distraction_time > alert_delay and not self.posture_alert:
                    self.posture_alert = True
            else:
                self.posture_distraction_time = None
                self.posture_alert = False

            if show_landmarks:
                self.drawer.draw_landmarks(
                    img,
                    pose_results.pose_landmarks,
                    self.pose_module.POSE_CONNECTIONS
                )
        else:
            posture_status = "Pose Not Found"
            posture_color = (0, 165, 255)
            self.posture_distraction_time = None
            self.posture_alert = False

        # -----------------------------
        # DRAW EYE POINTS
        # -----------------------------
        if face_results.multi_face_landmarks and show_landmarks:
            cv2.circle(img, (xaxis_left, yaxis_left), 3, (0, 255, 0), -1)
            cv2.circle(img, (xaxis_right, yaxis_right), 3, (0, 255, 0), -1)

        # -----------------------------
        # ALERT BANNERS
        # -----------------------------
        if self.eye_alert:
            cv2.rectangle(img, (10, 10), (450, 60), (0, 0, 255), -1)
            cv2.putText(img, "ALERT: You are distracted!", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if self.posture_alert:
            cv2.rectangle(img, (10, 70), (450, 120), (0, 0, 255), -1)
            cv2.putText(img, "ALERT: Incorrect posture!", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # -----------------------------
        # STATUS TEXT
        # -----------------------------
        cv2.putText(img, f"Eye Status: {eye_status}", (20, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)

        cv2.putText(img, f"Posture: {posture_status}", (20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)

        cv2.putText(img, f"Ear-Shoulder Offset: {int(posture_offset)} px", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="focus-posture-monitor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MonitorProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)