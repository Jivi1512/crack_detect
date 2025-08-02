import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st

model= load_model("model_alexnet.h5")
class_labels=["Not Cracked","Cracked"] # 0: cracked, 1: not cracked

st.title("Crack Detection App")
st.write("Upload an image to check for cracks.")

img=st.camera_input("Take a picture of the surface")
if img is not None:
    pil_img = Image.open(img)
    img_resize=pil_img.resize((224,224))
    img_array=image.img_to_array(img_resize)
    img_array=img_array/255.0
    img_array=np.expand_dims(img_array,axis=0)
    with st.spinner("Processing..."):
        prediction=model.predict(img_array)
        crack_probability=prediction.item()
        predicted_class = int(crack_probability > 0.5)
        confidence = crack_probability if predicted_class == 1 else 1 - crack_probability
        label= class_labels[predicted_class]
    st.success(f"Prediction: {label}\n")
    st.info(f"Confidence: {confidence*100:.2f} %")
    plt.imshow(pil_img)
    plt.title(f"{class_labels[predicted_class]} {confidence*100:.2f} %")
    plt.axis("off")
    plt.show()