import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Constants
DATA_DIR = r"C:\Users\divya\Downloads\archive\leaf_detection_dataset"
IMG_SIZE = 64

# Load images and labels
print("🔍 Loading dataset...")
X = []
y = []

categories = os.listdir(DATA_DIR)
print("Classes found:", categories)

for label, category in enumerate(categories):
    folder_path = os.path.join(DATA_DIR, category)
    for img_file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(img_gray.flatten())
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)
print(f"✅ Total images loaded: {len(X)}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
print("🧠 Training KNN classifier...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc * 100:.2f}%")
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Open file dialog to choose image
def choose_and_predict():
    root = tk.Tk()
    root.withdraw()  # hide root window
    file_path = filedialog.askopenfilename(
        title="Select a Leaf Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        print(f"\n🖼️ Selected image: {file_path}")
        predict_new_image(file_path)
    else:
        print("❌ No image selected.")

# Prediction function
def predict_new_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not load image.")
            return
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_flatten = img_gray.flatten().reshape(1, -1)
        prediction = model.predict(img_flatten)
        predicted_label = categories[prediction[0]]
        print("🌿 Predicted Disease:", predicted_label)
    except Exception as e:
        print("❌ Error:", e)

# Run the image selection and prediction
choose_and_predict()
