import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Page Configuration
st.set_page_config(page_title="Vehicle Detection Project", page_icon="🚗")

st.title("🚗 Vehicle Detection with YOLOv8")
st.write("Upload a traffic image to test the trained model.")

# 1. Load Model (Using cache to speed up)
@st.cache_resource
def load_model():
    # 1. Find the absolute path of this script (streamlit_app.py)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construct the model path relative to this script
    model_path = os.path.join(script_directory, 'trained_model.pt')
    
    # st.write(f"Searching for model at: {model_path}") # Debugging line

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Path searched: {model_path}")
        return None

model = load_model()

# 2. Image Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Detection Button
    if st.button("Detect Vehicles"):
        with st.spinner('Detecting...'):
            try:
                # Make Prediction (Keeping confidence low as requested)
                results = model(image, conf=0.15)
                
                # Plot results (returns BGR array)
                res_plotted = results[0].plot()
                
                # Convert from OpenCV BGR to RGB
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.success("Processing Complete!")
                st.image(res_rgb, caption='Detection Results', use_column_width=True)
                
                # Optional: Count detected objects
                boxes = results[0].boxes
                st.write(f"Total {len(boxes)} objects detected.")
                
            except Exception as e:
                st.error(f"Error during detection: {e}")

# Sidebar Info
st.sidebar.header("About Project")
st.sidebar.info(
    "This project is developed to detect traffic vehicles using YOLOv8.\n\n"
    "**Trained Classes:**\n"
    "- Car\n"
    "- Truck\n"
    "- Bus\n"
    "- Motorbike"
)