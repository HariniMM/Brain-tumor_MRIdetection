import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('best_model.keras')

class_names = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']

st.title(" Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan and this AI model will predict the type of tumor (or no tumor).")

uploaded_file = st.file_uploader(" Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    if st.button(" Predict Tumor Type"):
        #  Preprocess the image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        st.success(f"**Prediction:** {class_names[predicted_index]}")
        st.info(f" **Confidence:** {confidence:.2%}")
