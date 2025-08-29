import sqlite3
import os
from datetime import datetime

def init_db():
    # Create the database directory if it doesn't exist
    os.makedirs('database', exist_ok=True)
    # Connect to the database file in the database folder
    conn = sqlite3.connect('database/predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT,
                  predicted_digit INTEGER,
                  confidence REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_prediction(image_path, predicted_digit, confidence):
    # Create the database directory if it doesn't exist
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect('database/predictions.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO predictions (image_path, predicted_digit, confidence, timestamp) VALUES (?, ?, ?, ?)",
              (image_path, predicted_digit, confidence, timestamp))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()