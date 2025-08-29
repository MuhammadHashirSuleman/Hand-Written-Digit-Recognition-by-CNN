import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import seaborn as sns

RESULTS_DIR = 'results'
MODEL_PATH = 'model/saved_model/digit_recognizer.h5'

def preprocess_data():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    y_test_cat = tf.keras.utils.to_categorical(y_test)
    return x_test, y_test, y_test_cat

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

    x_test, y_test, y_test_cat = preprocess_data()

    # Predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_file = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report)
    print(f"Metrics saved to {metrics_file}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()

    # Line plot: per-class F1 scores
    from sklearn.metrics import precision_recall_fscore_support
    _, _, f1_scores, _ = precision_recall_fscore_support(y_test, y_pred, labels=range(10))
    plt.figure(figsize=(8, 5))
    plt.plot(range(10), f1_scores, marker='o', linestyle='-', color='b')
    plt.title('Per-Class F1 Scores')
    plt.xlabel('Digit')
    plt.ylabel('F1 Score')
    plt.xticks(range(10))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'f1_scores_line.png'))
    plt.close()

    print("Plots saved to results/")

if __name__ == '__main__':
    evaluate_model()
