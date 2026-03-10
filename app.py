import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pickle

# --- Page Config ---
st.set_page_config(page_title="Inference Engine", layout="wide")
st.title("Model Inference Dashboard")

# --- Model Caching ---
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_pkl_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
model_type = st.sidebar.radio("Select Model Type", ["YOLO", "Pickle Model"])

if model_type == "YOLO":
    model_path = st.sidebar.selectbox("Select Weights", [
        "models/best.pt", 
        "models/yolo11n.pt", 
        "models/yolo26n.pt"
    ])
    model = load_yolo_model(model_path)
else:
    model_path = "models/rice_detector.pkl"
    model = load_pkl_model(model_path)

# --- Main UI ---
uploaded_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Input Image", width='stretch')

    if st.button("Run Inference", type="primary"):
        with st.spinner("Processing..."):
            
            if model_type == "YOLO":
                # Execute YOLO prediction
                results = model(image)
                rendered_image = results[0].plot()
                
                st.subheader("Results")
                st.image(rendered_image, caption="Detection Output", width='stretch')
                
            elif model_type == "Pickle Model":
                # Execute Scikit-Learn/Pickle prediction
                img_array = np.array(image)
                
                # Update dimensions to match your exact training pipeline
                target_size = (224, 224) 
                
                # Preprocessing block
                img_resized = cv2.resize(img_array, target_size)
                img_flattened = img_resized.flatten().reshape(1, -1)
                
                # Inference
                prediction = model.predict(img_flattened)
                st.success(f"Classification Result: {prediction[0]}")