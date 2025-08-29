# Handwritten Digit Recognition

This project implements a handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. It includes a Flask-based web interface for users to draw or upload digit images and receive real-time predictions.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python model/train_model.py
   ```

3. **Initialize Database**:
   ```bash
   python database/db_manager.py
   ```

4. **Run the Flask App**:
   ```bash
   python app.py
   ```

5. Access the web interface at `http://localhost:5000`.

## Folder Structure
- `app.py`: Flask application for the web interface.
- `model/`: Contains scripts for training and prediction.
- `static/`: CSS, JavaScript, and uploaded images.
- `templates/`: HTML template for the web interface.
- `database/`: SQLite database handling.
- `utils/`: Image preprocessing functions.

## Requirements
- Python 3.8+
- TensorFlow, Flask, OpenCV, NumPy, Pillow
- MNIST dataset (automatically downloaded by TensorFlow)

## Features
- Draw or upload digit images for recognition.
- Real-time prediction with confidence scores.
- Stores predictions in an SQLite database.
- CNN model with high accuracy (>98% on MNIST test set, achieved 0.9960 on test evaluation).

## Visualization
![Test Results](results/test_results.png)
This image shows the predictions for test digits with true labels, predicted labels, and confidence scores.

## Notes
- The model is trained on the MNIST dataset (28x28 grayscale images).
- Ensure the `static/uploads/` directory exists before running the app.