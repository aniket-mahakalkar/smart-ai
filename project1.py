import streamlit as st
from PIL import Image
import google.generativeai as genai
import mediapipe as mp
import numpy as np
import time
import cv2

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

api_key = st.secrets["GOOGLE_API_KEY"]

def process_frame(frame, canvas, last_x, last_y):
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    save_gesture = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
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
    st.title("Virtual Calculator")
    st.image('Gemini AI.jpg')

    video_placeholder = st.empty()
    instructions_placeholder = st.empty()

    api_key = st.secrets["GOOGLE_API_KEY"]


    instructions_placeholder.text(
        "1. Use index finger for drawing.\n"
        "2. Use both index and middle finger for free movement.\n"
        "3. Use both index and thumb fingers together for erasing.\n"
        "4. Use three fingers up to save the image.\n"
        "5. The app will stop after saving the image."
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open video capture device.")
        return

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    canvas = None
    last_x, last_y = None, None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to read frame from capture device.")
            break

        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        frame, last_x, last_y, save_gesture = process_frame(frame, canvas, last_x, last_y)
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        if save_gesture:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = f'canvas_{timestamp}.png'
            cv2.imwrite(image_path, canvas)
            cap.release()
            cv2.destroyAllWindows()
            
            image = Image.open(image_path)
            # api_key = 'AIzaSyBZ6m7CmchPpMGGWIaovGLQ2g9eJj69Yg4'
            result = send_image_to_gemini(image, api_key)
            
            st.image(image)
            st.write("The output of given drawn problem is:", result)
            break

if __name__ == "__main__":
    main()
