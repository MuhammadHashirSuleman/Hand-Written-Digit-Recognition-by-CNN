import cv2
import numpy as np

def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image")
    
    # Inversion check (only if background is white)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Find bounding box
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("No digit found in image")
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize to 26x26 while preserving aspect ratio (closer to MNIST size)
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = 26
        new_h = int(26 / aspect_ratio)
    else:
        new_h = 26
        new_w = int(26 * aspect_ratio)
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Smoother interpolation

    # Put on 28x28 canvas, centered
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    # Debug: Save preprocessed image
    cv2.imwrite('debug_preprocessed.png', canvas)

    # Normalize to [0, 1]
    canvas = canvas.astype('float32') / 255.0
    canvas = np.expand_dims(canvas, axis=-1)
    canvas = np.expand_dims(canvas, axis=0)
    return canvas