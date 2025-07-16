import cv2
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
original_path=(r"C:\Users\agnih\Downloads\First Print-20250324T155521Z-001\First Print")
counterfeit_path=(r"C:\Users\agnih\Downloads\Second Print-20250324T155521Z-001\Second Print")
original_sample = cv2.imread(os.path.join(original_path, "input_image_active.png"), cv2.IMREAD_GRAYSCALE)
counterfeit_sample = cv2.imread(os.path.join(counterfeit_path, "input_image_assume.png"), cv2.IMREAD_GRAYSCALE)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(original_sample, cmap='gray')
ax[0].set_title("Original QR Code")
ax[1].imshow(counterfeit_sample, cmap='gray')
ax[1].set_title("Counterfeit QR Code")
plt.show()
def extract_edges(image):
    edges = cv2.Canny(image, 50, 150)
    return edges
original_edges = extract_edges(original_sample)
counterfeit_edges = extract_edges(counterfeit_sample)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(original_edges, cmap='gray')
ax[0].set_title("Edges - Original")
ax[1].imshow(counterfeit_edges, cmap='gray')
ax[1].set_title("Edges - Counterfeit")
plt.show()


def extract_lbp(image):
    lbp = local_binary_pattern(image, P=24, R=3, method='uniform')
    return lbp

original_lbp = extract_lbp(original_sample)
counterfeit_lbp = extract_lbp(counterfeit_sample)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(original_lbp, cmap='gray')
ax[0].set_title("LBP - Original")
ax[1].imshow(counterfeit_lbp, cmap='gray')
ax[1].set_title("LBP - Counterfeit")
plt.show()
from skimage.feature import hog

def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

hog_features_original = extract_hog_features(original_sample)
hog_features_counterfeit = extract_hog_features(counterfeit_sample)



####Prepare Dataset for model Training
## convert images into features and labels

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filepath in glob.glob(os.path.join(folder, "*.png")):  # Get full file path
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is not None:  # Ensure the image is loaded correctly
            img = cv2.resize(img, (128, 128))  # Resize for consistency
            images.append(img)
            labels.append(label)
    return images, labels

# Define dataset paths
original_path = r"C:\Users\agnih\Downloads\First Print-20250324T155521Z-001\First Print"
counterfeit_path = r"C:\Users\agnih\Downloads\Second Print-20250324T155521Z-001\Second Print"

# Load all images
original_images, original_labels = load_images_from_folder(original_path, 0)  # 0 = Original
counterfeit_images, counterfeit_labels = load_images_from_folder(counterfeit_path, 1)  # 1 = Counterfeit

# Combine data
X = np.array(original_images + counterfeit_images)  # Features
y = np.array(original_labels + counterfeit_labels)  # Labels

# Normalize images
X = X / 255.0  # Scale pixel values between 0 and 1

# Print dataset size
print("Total images loaded:", len(X))
print("Shape of dataset:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for deep learning model
X_train = X_train.reshape(-1, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)
# Flatten images for ML model
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)

# Evaluate
y_pred = svm_model.predict(X_test_flat)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

#### CNN MODEL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define CNN model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix

# SVM Model Evaluation
y_pred_svm = svm_model.predict(X_test_flat)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# CNN Model Evaluation
y_pred_cnn = (cnn_model.predict(X_test) > 0.5).astype("int32")
print("CNN Classification Report:\n", classification_report(y_test, y_pred_cnn))
import joblib

# Save the trained SVM model
joblib.dump(svm_model, "svm_qr_model.pkl")
print("SVM model saved successfully!")

# Save the trained CNN model
cnn_model.save("cnn_qr_model.h5")
print("CNN model saved successfully!")
