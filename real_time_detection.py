import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('asl_model.h5')


labels = np.load('labels.npy')

CONFIDENCE_THRESHOLD = 0.75

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size == 0:
                continue

            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized / 255.0
            reshaped = normalized.reshape(1, 64, 64, 1)

            preds = model.predict(reshaped, verbose=0)
            confidence = np.max(preds)
            class_id = np.argmax(preds)

            if confidence > CONFIDENCE_THRESHOLD:
                label = f"{labels[class_id]} ({confidence*100:.1f}%)"
            else:
                label = "Unknown"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )

    cv2.imshow("ASL Recognition (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

