# -*- coding: utf-8 -*-
'''
Real Time Face Registration with RFID Integration and LED Indication (No LCD)
'''

import cv2
import os
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time

# GPIO and LED Setup
LED_PIN = 18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# Setup RFID Reader
reader = SimpleMFRC522()

# Initialize Camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    while True:
        print("\n[INFO] Please scan your RFID card to begin...")
        try:
            rfid_id, rfid_text = reader.read()
            print(f"\n[INFO] RFID ID: {rfid_id}")
        except Exception as e:
            print(f"[ERROR] RFID Read Error: {e}")
            GPIO.cleanup()
            exit()

        # Ask for user's name
        name = input("Enter the name of the person: ").strip()

        # Create folder named after RFID if not exists
        rfid_folder = os.path.join('dataset', str(rfid_id))
        os.makedirs(rfid_folder, exist_ok=True)

        # Save name in a text file for future reference
        with open(os.path.join(rfid_folder, "name.txt"), "w") as f:
            f.write(name)

        print(f"\n[INFO] Registered name '{name}' for RFID {rfid_id}")
        print("[INFO] Initializing face capture. Look at the camera...")
        print("[INFO] LED ON during capture. Wait until it turns OFF.")

        count = 0
        GPIO.output(LED_PIN, GPIO.HIGH)

        while True:
            ret, img = cam.read()
            img = cv2.flip(img, -1)  # vertically flip
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                filename = os.path.join(rfid_folder, f"{count}.jpg")
                cv2.imwrite(filename, gray[y:y + h, x:x + w])
                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff
            if k == 27 or count >= 30:
                break

        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(1)
        print("[INFO] Face data saved successfully.\n")
        print("[INFO] Ready for next user. Press Ctrl+C to stop.\n")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user (Ctrl+C)")

finally:
    cam.release()
    cv2.destroyAllWindows()
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("[INFO] Cleanup complete. Exiting.")
