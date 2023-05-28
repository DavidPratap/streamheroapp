import streamlit as st
import os
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Dog and Cat Classifier using TensorFlow and Keras")


# Step1: Load the keras model and the image
model='cats_dogs_small_4.h5'
model=load_model(model)

#uploaded_file=st.file_uploader("Upload a cat or dog photo")
uploaded_file = st.file_uploader(
    "Choose your database", accept_multiple_files=False)
if uploaded_file is not None:
    file_name = uploaded_file
else:
    file_name = "image1.jpg"



# Step2 : get the file path and display the image
file_path=file_name


# Step 3 : Preprocess the image 
my_image=image.load_img(file_path, target_size=(150, 150))
my_img_arr=image.img_to_array(my_image)
my_img_arr=np.expand_dims(my_img_arr, axis=0)

if st.checkbox("Display Image", False):
    image=Image.open(uploaded_file)
    st.image(image)
# Step4: Get the prediction and print the result
prediction=int(model.predict(my_img_arr)[0][0])
if st.button("Predict"):
    if prediction==0:
        st.subheader("Its a Cat")
    if prediction==1:
        st.subheader('Its a dog')


