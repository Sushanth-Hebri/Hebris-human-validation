from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import io
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)

# Get the directory of the current file
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the Keras model
model_path = os.path.join(script_dir, 'keras_model.h5')
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    error_message = f'Error loading the model: {str(e)}'
    print(error_message)
    app.logger.error(error_message)
    raise

# Load the labels from the text file
labels_path = os.path.join(script_dir, 'labels.txt')
try:
    with open(labels_path, "r") as file:
        class_names = [name.strip() for name in file.readlines()]
except Exception as e:
    error_message = f'Error loading the labels: {str(e)}'
    print(error_message)
    app.logger.error(error_message)
    raise

# Function to preprocess the image
def preprocess_image(image_bytes, target_size=(224, 224)):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to RGB if it has 4 channels
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = image.resize(target_size)
        image = np.array(image)
        image = image / 255.0  # Normalize the image data
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        return jsonify({'error': f'Error preprocessing image: {str(e)}'})

# API endpoint for image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Check file extension
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'})

        # Read image bytes from the file
        image_bytes = file.read()

        # Preprocess the uploaded image
        image_data = preprocess_image(image_bytes)

        # Make prediction using the loaded model
        prediction = model.predict(image_data)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        
        # Check if predicted class is "male" or "female" and return "human"; otherwise return "other"
        if predicted_class_name in ['0 male', '1 female']:
            predicted_class_name = 'human'
        else:
            predicted_class_name = 'other'

        # Return predicted class as response
        return jsonify({'predicted_class': predicted_class_name})
    except Exception as e:
        error_message = f'Error predicting image: {str(e)}'
        print(error_message)
        app.logger.error(error_message)
        return jsonify({'error': error_message})

# Route for the / endpoint
@app.route('/', methods=['GET'])
def hello():
    return 'Hello'

if __name__ == '__main__':
    app.run(debug=True)
