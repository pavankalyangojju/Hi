import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
from PIL import Image

# GPIO Pins
BUZZER_PIN = 17
GREEN_LED_PIN = 26
RED_LED_PIN = 19
SERVO_PIN = 21

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)

# Servo setup
servo = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
servo.start(0)

# RFID and Face recognition setup
reader = SimpleMFRC522()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def move_servo():
    servo.ChangeDutyCycle(2.5)   # 0 degrees
    time.sleep(0.5)
    servo.ChangeDutyCycle(12.5)  # 180 degrees
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(3)
    servo.ChangeDutyCycle(2.5)   # back to 0 degrees
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

def get_images_and_labels(path):
    face_samples, ids = [], []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(path, file)).convert('L')
                img_np = np.array(img, 'uint8')
                faces = face_detector.detectMultiScale(img_np)
                for (x, y, w, h) in faces:
                    face_samples.append(img_np[y:y+h, x:x+w])
                    ids.append(1)
            except:
                continue
    return face_samples, ids

try:
    while True:
        print("\n[INFO] Please scan your RFID card...")
        try:
            rfid_id, _ = reader.read()
            rfid_id = str(rfid_id).strip()
            print(f"[INFO] RFID Scanned: {rfid_id}")
        except Exception as e:
            print(f"[ERROR] RFID Read Failed: {e}")
            continue

        image_folder = os.path.join("dataset", rfid_id)
        if not os.path.exists(image_folder):
            print(f"[ERROR] No dataset folder for RFID {rfid_id}")
            GPIO.output(RED_LED_PIN, GPIO.HIGH)
            time.sleep(2)
            GPIO.output(RED_LED_PIN, GPIO.LOW)
            continue

        print("[INFO] Training face recognizer...")
        faces, ids = get_images_and_labels(image_folder)
        if len(faces) == 0:
            print("[ERROR] No valid face data in folder.")
            continue

        recognizer.train(faces, np.array(ids))

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)
        print("[INFO] Look at the camera...")

        matched = False
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, -1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(detected_faces) != 1:
                print("[WARNING] No or multiple faces detected, skipping...")
                continue

            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y+h, x:x+w]
                id_pred, conf = recognizer.predict(face_roi)
                print(f"[INFO] Confidence: {conf:.2f}")

                if conf < 40:
                    print(f"[MATCH] Valid Face - Access Granted")

                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    time.sleep(2)
                    GPIO.output(GREEN_LED_PIN, GPIO.LOW)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)

                    move_servo()
                    matched = True
                    break
                else:
                    print("[WARNING] Unknown Face")
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    for _ in range(2):
                        GPIO.output(BUZZER_PIN, GPIO.HIGH)
                        time.sleep(0.3)
                        GPIO.output(BUZZER_PIN, GPIO.LOW)
                        time.sleep(0.3)
                    GPIO.output(RED_LED_PIN, GPIO.LOW)

            if matched or cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        time.sleep(3)

except KeyboardInterrupt:
    print("\n[INFO] Exiting Program.")
    servo.stop()
    GPIO.cleanup()
##hi
