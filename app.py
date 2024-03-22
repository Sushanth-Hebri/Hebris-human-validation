import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import streamlit as st

# Load the Keras model
model = load_model("keras_model.h5")

# Load the labels from the text file
with open("labels.txt", "r") as file:
    class_names = file.readlines()
    class_names = [name.strip() for name in class_names]

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image data
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Get the image filename from the user
image_name = st.text_input("Enter Image name", "sushanth-style.jpg")

# Preprocess the image
image_data = preprocess_image(image_name)

# Make prediction
prediction = model.predict(image_data)
predicted_class_index = np.argmax(prediction)
predicted_class_name = class_names[predicted_class_index]

# Map the predicted class name to 'human' or 'other'
if predicted_class_name in ['male', 'female']:
    predicted_class_name = 'human'
else:
    predicted_class_name = 'other'

# Display the image
st.image(image_name, caption="Input Image", use_column_width=True)

# Display the predicted class
st.write("Predicted Class:", predicted_class_name)



