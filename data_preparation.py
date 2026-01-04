import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATASET_PATH = r"D:\SIGN\asl_dataset"

LABELS = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S',
    'T','U','V','W','X','Y'
]

def load_images(dataset_path):
    images = []
    labels = []

    for label in LABELS:
        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (64, 64))
            img = img / 255.0                 
            img = img.reshape(64, 64, 1)      

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Load images
images, labels = load_images(DATASET_PATH)

# Convert labels → numbers using FIXED LABEL list
labels_numerical = np.array([LABELS.index(l) for l in labels])

# One-hot encode
labels_categorical = to_categorical(labels_numerical, num_classes=len(LABELS))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, random_state=42
)

# Save files
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("labels.npy", np.array(LABELS)) 
