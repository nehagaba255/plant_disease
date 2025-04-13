import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("plant_model.keras")

# Class labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

st.markdown("""
    <style>
        body {
            background-color: #e6ffe6;
        }
        .stApp {
            background-color: #e6ffe6;
        }
        .title {
            font-size: 42px;
            color: #2e7d32;
            text-align: center;
            font-weight: bold;
        }
        .subtitle {
            font-size: 20px;
            color: #1b5e20;
            text-align: center;
        }
        .prediction {
            font-size: 22px;
            font-weight: bold;
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

# UI 
st.markdown("<div class='title'>🌿 Plant Disease Detection App 🌿</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a leaf image to identify possible plant disease</div><br>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.markdown(f"<div class='prediction'> Prediction: <i>{predicted_class}</i></div>", unsafe_allow_html=True)
else:
    st.info("Please upload a leaf image to get a prediction.")
