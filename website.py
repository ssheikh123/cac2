import streamlit as st
from PIL import Image
import cv2
from datetime import datetime
import numpy as np
import requests
import base64
import creds
from dotenv import load_dotenv
import os

st.title("Camera")
st.header("This is a camera", divider="rainbow")

# Initialize session state for capturing the photo
if "photo_taken" not in st.session_state:
    st.session_state["photo_taken"] = False
if "photo" not in st.session_state:
    st.session_state["photo"] = None
if "take_photo_clicked" not in st.session_state:
    st.session_state["take_photo_clicked"] = False  # New state to check if photo button was clicked
if "return_to_camera" not in st.session_state:
    st.session_state["return_to_camera"] = False  # New state to track return to camera button visibility
if "api_response" not in st.session_state:
    st.session_state["api_response"] = None  # State to store the API response

# OpenAI API Key
api_key = os.getenv("api_key")

# Function to encode the image to base64
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Function to open the camera and display the preview
def live_camera_preview():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()  # Placeholder for the video stream

    # Create the "Take Photo" button outside the loop
    take_photo_button = st.button("Take Photo")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not get frame from the camera.")
            break

        # Convert the frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the placeholder
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Check if the "Take Photo" button was clicked
        if take_photo_button:  # Check the button click
            st.session_state["photo"] = frame_rgb  # Save the captured frame to session state
            st.session_state["photo_taken"] = True  # Set the photo taken flag
            st.session_state["return_to_camera"] = True  # Show the "Return to Camera" button
            cap.release()
            frame_placeholder.empty()  # Clear the live feed
            break  # Stop the loop once the photo is taken

# Function to send the image to the OpenAI API for analysis
def send_image_to_openai(image):
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Payload as per the second code sample
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a picture of food. Return only a string of comma separated values based on an approximate serving size estimate. If there are multiple food items, return multiple strings. follow this format: Food item (include brand if applicable) (string), calories (int), sugar, fat, protein, iron, carbohydrates. If there is no observable food in the picture, respond with No food detected. Here is an example of a response: Yogurt, *all its nutrition info* \n Jelly Beans, *all its nutrition info* You should return nothing else besides these strings."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # Send the request to the API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Check the response status
    if response.status_code == 200:
        response_data = response.json()
        message_content = response_data['choices'][0]['message']['content']
        return message_content
    else:
        st.error(f"Failed to get response from API: {response.status_code}")
        return None

# Start the live camera preview if no photo is taken
if not st.session_state["photo_taken"]:
    live_camera_preview()

# If a photo was taken, display and save it
if st.session_state["photo_taken"]:
    st.image(st.session_state["photo"], caption="Captured Photo")

    # Save the image with a timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'photo_{timestamp}.png'
    
    cv2.imwrite(file_name, cv2.cvtColor(st.session_state["photo"], cv2.COLOR_RGB2BGR))  # Convert back to BGR to save

    st.success(f"Photo saved as {file_name}")

    # Send the image to the OpenAI API and display the response
    if st.button("Analyze Image"):
        st.session_state["api_response"] = send_image_to_openai(st.session_state["photo"])
    
    if st.session_state["api_response"]:
        st.write("API Response:")
        st.write(st.session_state["api_response"])

    # Only show the "Return to Camera" button if we are in the photo display mode
    if st.session_state["return_to_camera"]:
        if st.button("Return to Camera"):
            # Reset session state to return to live camera feed
            st.session_state["photo_taken"] = False
            st.session_state["photo"] = None
            st.session_state["take_photo_clicked"] = False  # Reset the photo button flag
            st.session_state["return_to_camera"] = False  # Hide the "Return to Camera" button
            st.session_state["api_response"] = None  # Clear the API response
            st.cache_data.clear()  # This will clear any cached state
            st.rerun()  # Force a rerun to restart the camera feed
