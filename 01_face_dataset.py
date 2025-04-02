import cv2
import os
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time

# GPIO Setup
LED_PIN = 18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# Setup RFID Reader
reader = SimpleMFRC522()

# Initialize Camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    while True:
        print("\n[INFO] Please scan your RFID card to begin...")
        try:
            rfid_id, _ = reader.read()
            print(f"[INFO] RFID ID: {rfid_id}")
        except Exception as e:
            print(f"[ERROR] RFID Read Error: {e}")
            GPIO.cleanup()
            exit()

        name = input("Enter the name of the person: ").strip()

        rfid_folder = os.path.join('dataset', str(rfid_id))
        os.makedirs(rfid_folder, exist_ok=True)

        with open(os.path.join(rfid_folder, "name.txt"), "w") as f:
            f.write(name)

        print(f"[INFO] Registered name '{name}' for RFID {rfid_id}")
        print("[INFO] Initializing face capture. Look at the camera...")
        print("[INFO] LED ON during capture. Wait until it turns OFF.")

        count = 0
        GPIO.output(LED_PIN, GPIO.HIGH)

        while True:
            ret, img = cam.read()
            if not ret or img is None:
                print("[ERROR] Camera read failed. Retrying...")
                continue

            img = cv2.flip(img, -1)
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
        print("[INFO] Capture complete.\nReady for next user or Ctrl+C to exit.")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Exiting.")

finally:
    cam.release()
    cv2.destroyAllWindows()
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("[INFO] Cleanup complete.")
#######HI
