import streamlit as st
from PIL import Image
import google.generativeai as genai
import mediapipe as mp
import numpy as np
import time
import cv2

# Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set page config
st.set_page_config(layout="centered")

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'last_x' not in st.session_state:
    st.session_state.last_x = None
if 'last_y' not in st.session_state:
    st.session_state.last_y = None
if 'hands' not in st.session_state:
    st.session_state.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def process_frame(frame, canvas, last_x, last_y):
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = st.session_state.hands.process(image_rgb)
    save_gesture = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            index_finger_extended = index_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            middle_finger_extended = middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            thumb_extended = thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y

            if index_finger_extended and not middle_finger_extended:
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (ix, iy), (225, 225, 225), 5)
                last_x, last_y = ix, iy
            elif index_finger_extended and middle_finger_extended:
                last_x, last_y = ix, iy
            else:
                last_x, last_y = None, None

            if index_finger_extended and thumb_extended and last_x is not None and last_y is not None:
                if abs(ix - tx) < 30 and abs(iy - ty) < 30:
                    cv2.line(canvas, (last_x, last_y), (ix, iy), (0, 0, 0), 50)
                last_x, last_y = ix, iy

            extended_fingers_count = sum(
                hand_landmarks.landmark[f].y < hand_landmarks.landmark[f - 2].y
                for f in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                          mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
            )

            if extended_fingers_count == 3:
                save_gesture = True

    combined = cv2.addWeighted(frame, 1, canvas, 1, 0)
    return combined, last_x, last_y, save_gesture

def send_image_to_gemini(image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(['Solve the given mathematical problem in the image in detail and also explain it', image])
    return response.text

def main():
    st.title("✍️ Virtual Calculator with Gemini")
    st.image("Gemini AI.jpg")

    st.markdown(
        "**Instructions:**\n"
        "1. Use index finger to draw.\n"
        "2. Use index + middle finger to move.\n"
        "3. Use index + thumb to erase.\n"
        "4. Raise 3 fingers to save and analyze.\n"
    )

    if st.button("Start" if not st.session_state.running else "Stop"):
        if not st.session_state.running:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.running = True
        else:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.running = False
            st.session_state.canvas = None
            st.experimental_rerun()

    video_placeholder = st.empty()

    if st.session_state.running and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            if st.session_state.canvas is None:
                st.session_state.canvas = np.zeros_like(frame)

            frame, st.session_state.last_x, st.session_state.last_y, save_gesture = process_frame(
                frame, st.session_state.canvas, st.session_state.last_x, st.session_state.last_y
            )

            video_placeholder.image(frame, channels="BGR")

            if save_gesture:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_path = f"canvas_{timestamp}.png"
                cv2.imwrite(image_path, st.session_state.canvas)
                st.session_state.cap.release()
                st.session_state.running = False
                st.session_state.cap = None

                st.image(image_path, caption="Captured Image")
                st.success("Image saved! Sending to Gemini...")

                image = Image.open(image_path)
                api_key = ""  # Replace with actual key
                result = send_image_to_gemini(image, api_key)
                st.write("🔍 Gemini Output:")
                st.write(result)
                st.stop()

        st.experimental_rerun()

if __name__ == "__main__":
    main()
