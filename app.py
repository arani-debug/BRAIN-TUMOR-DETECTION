import streamlit as st
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import gdown
import base64

# Set page title
st.set_page_config(page_title="Brain Tumor Detection")

# Google Drive Model Link
MODEL_URL = "https://drive.google.com/uc?id=13di0px10kBfKqgdaI6B8aIFeoHsAmVhb"
MODEL_PATH = "model.h5"

# Function to set a background image
def set_background(image_path):
    with open(image_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1, h3, h4, p, label {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Set background image (Replace "background.jpg" with your image file)
set_background("background.jpg")

# Function to download model safely
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model... This may take a few minutes ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    # Check if the file was downloaded correctly
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        st.success("")
    else:
        st.error("Model download failed. Please check your internet connection.")

# Download and load model
download_model()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Function to predict brain tumor
def predict_tumor(image):
    image = image.resize((128, 128))  # Resize to match model input
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index]}"

    return result, confidence_score

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: white;'> Brain Tumor Detection </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: white;'>Upload an MRI scan to check for brain tumor presence.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if "model" in globals():
        prediction, confidence = predict_tumor(image)
        st.markdown(f"<h3 style='color: white;'>Prediction: {prediction}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: white;'>Confidence: {confidence * 100:.2f}%</h4>", unsafe_allow_html=True)
    else:
        st.error("Model is not loaded. Please refresh and try again.")
