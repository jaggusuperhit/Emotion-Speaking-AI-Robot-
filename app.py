import streamlit as st
import cv2
import google.generativeai as genai
import pyttsx3
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# api configuration (gemini api)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# initialize text to speech engine
tts_engine = pyttsx3.init()

# Custom functions
def generate_image_response(image):
    """Generate response based on the image"""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    prompt = "Analyze the emotions conveyed by the person in this image and respond directly to them, using 'you' in a short and empathetic way."
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content([prompt, img])
    return response.text

def speak_response(response_text):
    """Speak the response"""
    tts_engine.say(response_text)
    tts_engine.runAndWait()

# App
st.title("Emotional Intelligent Robot")
st.subheader("It will detect your emotions from live video and guide you")

# Working with camera
camera = cv2.VideoCapture(0)

# Create a container for the video feed
video_container = st.empty()

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        st.write("Unable to capture video")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_container.image(frame_rgb, channels='RGB')

    response = generate_image_response(frame)

    if response:
        st.write(response)
        speak_response(response)

    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()