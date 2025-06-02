import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('saved_model/mask_detector_model.h5')
categories = ['With Mask', 'Without Mask', 'Improper Mask']

input_shape = model.input_shape[1:3]

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
if face_cascade.empty():
    print("Failed to load face cascade classifier xml file!")
    exit(1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray is None or gray.size == 0:
        print("Empty grayscale frame!")
        continue

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, input_shape)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        class_index = np.argmax(prediction)

        label = categories[class_index]
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow('Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
