from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
try:
    model = tf.keras.models.load_model('model/saved_model/digit_recognizer.h5')
    print("Model loaded successfully at 07:46 PM PKT, August 29, 2025")
except Exception as e:
    print(f"Error loading model at 07:46 PM PKT, August 29, 2025: {str(e)}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received /predict request at 07:46 PM PKT, August 29, 2025")
        # Handle image file upload
        if 'image' in request.files:
            print("Processing file upload")
            file = request.files['image']
            if file.filename == '':
                print("Error: No file selected")
                return jsonify({'error': 'No file selected'}), 400

            # Save uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_digit.png')
            file.save(filepath)
            print(f"Image saved to: {filepath}")

            # Preprocess image using updated preprocess.py
            processed_image = preprocess_image(filepath)
            print(f"Preprocessed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")
        
        else:
            print("Error: No valid image or data provided")
            return jsonify({'error': 'No valid image or data provided'}), 400

        # Verify model is loaded
        if model is None:
            print("Error: Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500

        # Verify input shape and values
        print(f"Processed image shape: {processed_image.shape}, Max: {processed_image.max()}, Min: {processed_image.min()}")

        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        print(f"Raw prediction: {prediction}")
        digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][digit])
        print(f"Predicted digit: {digit}, Confidence: {confidence}")

        # Handle low confidence
        message = None
        if confidence < 0.5:
            message = "Uncertain prediction, please upload a better image"

        # Save to database (if available)
        try:
            from database.db_manager import save_prediction
            filepath = filepath if 'filepath' in locals() else 'file_input'
            save_prediction(filepath, digit, confidence)
            print("Prediction saved to database")
        except ImportError:
            print("Warning: save_prediction not available, skipping database save")

        return jsonify({
            'digit': digit,
            'confidence': confidence * 100,  # Convert to percentage
            'message': message,
            'probabilities': [float(p) for p in prediction[0]]  # For debugging
        })

    except Exception as e:
        print(f"Error during prediction at 07:46 PM PKT, August 29, 2025: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print(f"Starting Flask app with UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']} at 07:46 PM PKT, August 29, 2025")
    app.run(debug=True)