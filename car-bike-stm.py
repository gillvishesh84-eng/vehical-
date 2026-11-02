
import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Vehicle Classification AI",
    page_icon="🚗",
    layout="centered"
)

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("CAT.keras")

model = load_model()

# App title
st.title("🚗 Vehicle Classification AI")
st.write("Upload an image to identify the vehicle.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Predict button
    if st.button("Predict Vehicle"):

        # Resize image
        img = img.resize((64, 64))

        # Convert image to array
        x = tf.keras.utils.img_to_array(img)

        # Normalize only if training used rescaling to 0-1
        x = np.expand_dims(x, axis=0)

        # Prediction
        prediction = model.predict(x)

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Show result
        st.success(
            f"Predicted Vehicle: {class_names[predicted_index]}"
        )

        st.metric("Confidence", f"{confidence:.2f}%")
