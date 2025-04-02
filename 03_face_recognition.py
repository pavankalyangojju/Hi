import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
from PIL import Image
from datetime import datetime
from collections import defaultdict

# GPIO Setup
BUZZER_PIN = 17
GREEN_LED_PIN = 26
RED_LED_PIN = 19
SERVO_PIN = 21

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

attendance_log = defaultdict(list)

def move_servo():
    servo.ChangeDutyCycle(2.5)  # 0 degrees
    time.sleep(0.5)
    servo.ChangeDutyCycle(12.5)  # 180 degrees
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(3)
    servo.ChangeDutyCycle(2.5)  # back to 0 degrees
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

reader = SimpleMFRC522()
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

try:
    while True:
        print("\n[INFO] Please scan your RFID card...")

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        try:
            rfid_id, _ = reader.read()
            rfid_id = str(rfid_id)
            print(f"[INFO] RFID Scanned: {rfid_id}")
        except Exception as e:
            print(f"[ERROR] RFID Read Failed: {e}")
            GPIO.cleanup()
            break

        image_folder = os.path.join("dataset", rfid_id)
        if not os.path.exists(image_folder):
            print(f"[ERROR] No dataset folder found for RFID {rfid_id}")
            GPIO.output(RED_LED_PIN, GPIO.HIGH)
            time.sleep(2)
            GPIO.output(RED_LED_PIN, GPIO.LOW)
            continue

        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
            face_samples = []
            ids = []
            for image_path in image_paths:
                try:
                    img = Image.open(image_path).convert('L')
                    img_np = np.array(img, 'uint8')
                    faces = face_detector.detectMultiScale(img_np)
                    for (x, y, w, h) in faces:
                        face_samples.append(img_np[y:y+h, x:x+w])
                        ids.append(1)
                except:
                    continue
            return face_samples, ids

        print("[INFO] Training model...")
        faces, ids = get_images_and_labels(image_folder)
        recognizer.train(faces, np.array(ids))

        print("[INFO] Look at the camera...")
        matched = False

        while True:
            ret, img = cam.read()
            if not ret:
                continue

            img = cv2.flip(img, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 40:
                    name_file = os.path.join(image_folder, "name.txt")
                    if os.path.exists(name_file):
                        with open(name_file, "r") as f:
                            person_name = f.read().strip()
                    else:
                        person_name = "Matched"

                    print(f"[INFO] Face matched - {person_name}")
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    time.sleep(2)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

                    move_servo()
                    matched = True
                    break
                else:
                    print("[WARNING] Unknown face detected")
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    for _ in range(2):
                        GPIO.output(BUZZER_PIN, GPIO.HIGH)
                        time.sleep(0.3)
                        GPIO.output(BUZZER_PIN, GPIO.LOW)
                        time.sleep(0.3)
                    GPIO.output(RED_LED_PIN, GPIO.LOW)

            cv2.imshow("camera", img)
            if matched or cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        time.sleep(3)
###Hi
except KeyboardInterrupt:
    print("\n[INFO] Program interrupted. Exiting gracefully.")
    servo.stop()
    GPIO.cleanup()
