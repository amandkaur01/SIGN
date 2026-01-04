import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("asl_model.h5")

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Error Rate: {(1 - accuracy) * 100:.2f}%")
