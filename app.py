import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import gdown
import os

# --- Step 1: Download model from Google Drive if not exists ---
file_id = "1yQ_oqEvwVqo09Txs8gUWfMZxtlwGqZJ9"  # Your model file ID
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "pneumonia_detector.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(url, model_path, quiet=False)

# --- Step 2: Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- Step 3: App UI ---
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image, and the model will predict if it's **Normal** or **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

    st.subheader(f"Prediction: **{result}**")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
