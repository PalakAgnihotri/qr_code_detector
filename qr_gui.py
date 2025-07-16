import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from PIL import Image, ImageTk
import joblib
from tensorflow.keras.models import load_model

# Load trained models
svm_model = joblib.load("svm_qr_model.pkl")  # Load SVM model
cnn_model = load_model("cnn_qr_model.h5")    # Load CNN model

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize
    img = img / 255.0  # Normalize
    return img

# Function to predict QR code authenticity
def predict_qr_code(image_path):
    processed_img = preprocess_image(image_path)

    # SVM Prediction
    svm_input = processed_img.flatten().reshape(1, -1)
    svm_pred = svm_model.predict(svm_input)[0]

    # CNN Prediction
    cnn_input = processed_img.reshape(1, 128, 128, 1)
    cnn_pred = cnn_model.predict(cnn_input)[0]
    cnn_label = 1 if cnn_pred > 0.5 else 0

    # Compare results
    if svm_pred == cnn_label:
        result = "Original" if cnn_label == 0 else "Counterfeit"
    else:
        result = "Uncertain (SVM & CNN Disagree)"
    
    return result

# Function to select image and classify QR code
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        # Load image in Tkinter window
        img = Image.open(file_path)
        img = img.resize((200, 200))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)

        # Display image
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # Predict result
        result = predict_qr_code(file_path)

        # Update label with prediction
        label_result.config(text=f"Prediction: {result}", font=("Arial", 14, "bold"))

# Create Tkinter window
root = tk.Tk()
root.title("QR Code Authentication")
root.geometry("400x500")

# UI Elements
label_title = Label(root, text="QR Code Detector", font=("Arial", 16, "bold"))
label_title.pack(pady=10)

btn_upload = Button(root, text="Upload QR Code", command=open_file, font=("Arial", 12))
btn_upload.pack(pady=10)

label_image = Label(root)
label_image.pack(pady=10)

label_result = Label(root, text="", font=("Arial", 14))
label_result.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
