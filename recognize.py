import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("asl_model.h5")

labels = np.load("labels.npy")  

cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.60

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    # ROI box (keep hand inside this)
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 64, 64, 1)

    # Predict
    preds = model.predict(reshaped, verbose=0)
    confidence = np.max(preds)
    class_id = np.argmax(preds)

    if confidence > CONFIDENCE_THRESHOLD:
        text = f"{labels[class_id].upper()} ({confidence*100:.1f}%)"
    else:
        text = "Unknown"

    # Display text
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
