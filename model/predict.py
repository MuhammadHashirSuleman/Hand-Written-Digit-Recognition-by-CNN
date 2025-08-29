import tensorflow as tf
import numpy as np

def predict_digit(image):
    model = tf.keras.models.load_model('model/saved_model/digit_recognizer.h5')
    prediction = model.predict(image)
    digit = np.argmax(prediction[0])
    confidence = float(prediction[0][digit])
    return digit, confidence