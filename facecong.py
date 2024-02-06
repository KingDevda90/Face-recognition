import cv2
import streamlit as st
#import face_recognition
#from PIL import Image,ImageDraw
#import numpy as np

face_cascade = cv2.CascadeClassifier('/Users/mac/Desktop/GoMyCode/haarcascade_frontalface_default.xml')

def detect_faces_in_video(color, min_neighbors, scale_factor, save_image_checkbox):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Read the frames from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frames using Streamlit
    st.image(frame, channels="BGR")

    # Save image if checkbox is checked
    if save_image_checkbox:
        cv2.imwrite("detected_faces.jpg", frame)

    # Release the webcam
    cap.release()

def detect_faces_in_photo(image_path, color, min_neighbors, scale_factor, save_image_checkbox):
    # Load the image from the file
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Display the image with rectangles using Streamlit
    st.image(image, channels="BGR")

    # Save image if checkbox is checked
    if save_image_checkbox:
        cv2.imwrite("detected_faces.jpg", image)

def app():
    st.title("Face Detection")

    # Instructions
    st.write("Welcome to the Face Detection App! Press the button below to start detecting faces from your webcam or upload a photo.")
    st.write("Adjust the parameters and customize your experience.")

    # User input for rectangle color
    color = st.color_picker("Choose rectangle color", "#00ff00")

    # User input for minNeighbors
    min_neighbors = st.slider("Select minNeighbors", 1, 10, 5)

    # User input for scaleFactor
    scale_factor = st.slider("Select scaleFactor", 1.1, 2.0, 1.3, step=0.1)

    # Checkbox pour l'enregistrement de l'image
    save_image_checkbox = st.checkbox("Save Image")

    # Option to choose between webcam and photo upload
    option = st.radio("Choose Input Source", ("Webcam", "Upload Photo"))

    if option == "Webcam":
        if st.button("Start Detection"):
            # Call the detect_faces_in_video function with user-selected parameters
            detect_faces_in_video(color, min_neighbors, scale_factor, save_image_checkbox)
    else:
        # File uploader for photo
        uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            if st.button("Start Detection"):
                # Call the detect_faces_in_photo function with user-selected parameters
                detect_faces_in_photo(uploaded_file, color, min_neighbors, scale_factor, save_image_checkbox)

if __name__ == "__main__":
    app()
