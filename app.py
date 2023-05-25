import streamlit as st
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Dog and Cat Classifier using TensorFlow and Keras")
model='cats_dogs_small_2.h5'

# Step1: Load the keras model and the image
model=load_model(model)
from tensorflow.keras.preprocessing import image
uploaded_file=st.file_uploader("Upload a cat or dog photo")
if uploaded_file is not None:
    file_path = uploaded_file.name
    st.write(file_path)
else:
    path_in = None

# Step2 : get the file path 
# if uploaded_file:
#    st.write("Filename: ", uploaded_file.name)

# for file in uploaded_file:
#     file_path=uploaded_file.name

# Step 3 : Preprocess the image 
my_image=image.load_img(file_path, target_size=(150, 150))
my_img_arr=image.img_to_array(my_image)
if st.checkbox("Display Image", False):
    image=Image.open(uploaded_file)
    st.image(image)
my_img_arr=np.expand_dims(my_img_arr, axis=0)


# Step4: Get the prediction and print the result
prediction=int(model.predict(my_img_arr)[0][0])
if st.button("Predict"):
    if prediction==0:
        st.write("Its a Cat")
    if prediction==1:
        st.write('Its a dog')

# image = tensorflow.keras.utils.load_img(file_path)
# input_arr = tf.keras.utils.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.
# predictions = model.predict(input_arr)
