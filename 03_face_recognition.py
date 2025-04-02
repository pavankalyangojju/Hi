import cv2
import numpy as np
import os

# Load trained face recognizer and Haar cascade for detection
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font for labeling
font = cv2.FONT_HERSHEY_SIMPLEX

# Names mapped to IDs (update these as per your training)
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

# Start camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Minimum face size to recognize
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, -1)  # flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_pred, confidence = recognizer.predict(roi_gray)

        if confidence < 40:
            name = names[id_pred] if id_pred < len(names) else "Unknown"
            id_text = name
            box_color = (0, 255, 0)  # Green for matched
        else:
            id_text = "Unknown Face"
            box_color = (0, 0, 255)  # Red for unknown

        confidence_text = "  {0}%".format(round(100 - confidence))

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(img, id_text, (x + 5, y - 5), font, 1, box_color, 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:  # ESC key to exit
        break

print("\n[INFO] Exiting program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
