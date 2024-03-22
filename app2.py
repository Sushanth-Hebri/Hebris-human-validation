from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Define custom objects for loading the model
custom_objects = {'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}

# Load the Keras model using custom objects
model = tf.keras.models.load_model('keras_model.h5', custom_objects=custom_objects)

# Load the labels from the text file
with open("labels.txt", "r") as file:
    class_names = file.readlines()
    class_names = [name.strip() for name in class_names]

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    
    # Convert image to RGB if it has 4 channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image data
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Render the HTML form for image upload
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# API endpoint for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Check file extension
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'})

    image_path = 'uploaded_image.jpg'
    file.save(image_path)

    # Preprocess the uploaded image
    image_data = preprocess_image(image_path)

    # Make prediction using the loaded model
    prediction = model.predict(image_data)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    # Check if predicted class is "male" or "female" and return "human"; otherwise return "other"
    if predicted_class_name.lower() in ['male', 'female']:
        predicted_class_name = 'human'
    else:
        predicted_class_name = 'other'

    # Return predicted class as response
    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
