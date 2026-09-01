import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray scan to classify it as **Normal** or **Pneumonia**.")

# Cache model loading for performance
@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model('best_model.h5')

try:
    model = load_pneumonia_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Image preprocessing helper
def preprocess_xray(image):
    # PneumoniaMNIST requires 28x28 grayscale
    image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 28, 28, 1)
    return img_array

uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Scan")
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, use_container_width=True)
        
    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_xray(raw_image)
            prediction = model.predict(processed_img)[0][0]
            
            if prediction > 0.5:
                st.error("**Pneumonia Detected**")
                st.metric("Confidence", f"{prediction * 100:.2f}%")
            else:
                st.success("**Normal (No Pneumonia)**")
                st.metric("Confidence", f"{(1 - prediction) * 100:.2f}%")

st.divider()
st.caption("⚠️ **Disclaimer:** Educational demo only. Not intended for clinical diagnostic use.")