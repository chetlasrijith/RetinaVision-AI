import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# Load the model
model_path = os.path.join(os.path.dirname(__file__), "stundent_model.keras")
model = tf.keras.models.load_model(model_path)

# Preprocess function (grayscale + resize + equalizeHist)
def preprocess_image(uploaded_image):
    # Convert to RGB and resize
    img = uploaded_image.convert("RGB")
    img = img.resize((224, 224))

    # Convert to numpy array and normalize
    img_array = np.array(img).astype("float32") / 255.0

    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    return img_array


# Class labels (update with your actual class names if available)
CLASS_NAMES = [
    'AMD',                    # 0
    'Cataract',               # 1
    'DR',                     # 2
    'Glaucoma',               # 3
    'Hypertensive_Retinopathy', # 4
    'Normal_Fundus',          # 5
    'Pathological_Myopia'     # 6
]



# Streamlit UI
st.title("Retinal Disease Classification")
st.write("Upload a retinal fundus image to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.subheader("Prediction")
    st.write(f"Predicted Class: **{CLASS_NAMES[predicted_class]}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
