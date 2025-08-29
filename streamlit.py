import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Loading the pre trained model
model = load_model("mobilenet.keras")

class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]

# Image preprocessing function
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB") 
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Confidence checking
def validate_confidence(confidence_score: float, threshold: float = 0.9):
    if confidence_score < threshold:
        st.error(f"Low confidence: {confidence_score*100:.2f}%. Please upload a another image or Contact the Developer.")
        return False
    return True

#Streamlit UI
st.title("Blood Cell Cancer Detection")
st.write("Upload the microscopic image of the blood cell.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)  

    # Preprocessing the uploaded image and predicting its class
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(np.max(predictions))

    # Validating the confidence score
    if validate_confidence(confidence):
        st.markdown(f"**Prediction:** {predicted_class_name}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
