"""
Smart Security System - Laptop Side (Fixed ভার্সন)
------------------------------------
এই ভার্সনে যা ফিক্স করা হয়েছে:
  1. Training আর Recognition দুই জায়গাতেই face একই size-এ resize করা হয় +
     histogram equalization করা হয় (lighting পার্থক্যের কারণে "চেনা মানুষ অচেনা"
     সমস্যা কমানোর জন্য)।
  2. Debug মোডে প্রতিটা face-এর confidence value print হয়, যাতে তুমি নিজের
     dataset অনুযায়ী সঠিক CONFIDENCE_THRESHOLD ঠিক করতে পারো।
  3. Webcam খোলার সময় CAP_DSHOW backend ব্যবহার করা হয় (MSMF error এড়াতে)।
  4. detectMultiScale-এ minSize যোগ করা হয়েছে, যাতে ছোট/ভুল region face হিসেবে
     ধরা না পড়ে।

প্রয়োজনীয় লাইব্রেরি:
    pip install opencv-contrib-python requests numpy

ফোল্ডার স্ট্রাকচার (আগের মতোই):
    dataset/
        person_name_1/
        person_name_2/
    unknown_faces/
"""

import cv2
import os
import time
import requests
import numpy as np

# ---------------------- CONFIG ----------------------
ESP32_IP = "192.168.1.170"
ESP32_OPEN_URL = f"http://{ESP32_IP}/door_open"
ESP32_ALERT_URL = f"http://{ESP32_IP}/unknown_alert"
ESP32_MOTION_URL = f"http://{ESP32_IP}/motion_status"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_SCRIPT_DIR, "..", "dataset")
UNKNOWN_DIR = os.path.join(_SCRIPT_DIR, "..", "unknown_faces")

# ⚠️ এটাই মূল টিউনিং প্যারামিটার। DEBUG_MODE = True রেখে আগে actual confidence
# value গুলো দেখে নাও (known vs unknown), তারপর এই মান ঠিক করো।
# CLAHE preprocessing-এর পর Abir-এর confidence অনেক স্থিতিশীল (৯৯.৫-১০২.৪),
# তাই threshold তার একটু উপরে বসানো হলো।
CONFIDENCE_THRESHOLD = 110

REQUIRED_CONSECUTIVE_MATCHES = 1   # ১ বার known হলেই এখন দরজা খুলবে (false-positive protection কমে গেল)
MOTION_POLL_INTERVAL = 1.0
RECOGNITION_WINDOW_SECONDS = 8
COOLDOWN_AFTER_TRIGGER = 5

FACE_SIZE = (200, 200)   # training আর recognition দুই জায়গাতেই এই সাইজে resize হবে
DEBUG_MODE = True        # terminal-এ confidence value print হবে কিনা

# ল্যাপটপের বিল্ট-ইন ক্যামেরা সাধারণত 0, external webcam সাধারণত 1 বা তার বেশি।
# find_camera_index.py চালিয়ে সঠিক নাম্বারটা বের করে এখানে বসাও।
CAMERA_INDEX = 0
# ------------------------------------------------------

os.makedirs(UNKNOWN_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# grid_x/grid_y বাড়িয়ে finer local detail ধরা হচ্ছে, যাতে দুইজনের মুখের
# পার্থক্য background-এর চেয়ে বেশি গুরুত্ব পায় (default grid_x=8, grid_y=8)
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=10, grid_y=10)


# CLAHE (Contrast Limited Adaptive Histogram Equalization) - সাধারণ equalizeHist()
# এর চেয়ে আলোর তারতম্য অনেক ভালোভাবে সামলায়, তাই confidence value বেশি স্থিতিশীল থাকে
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face(gray_face):
    """Training আর recognition দুই জায়গাতেই একই preprocessing হওয়া জরুরি,
    নাহলে lighting/size পার্থক্যের কারণে চেনা মানুষও অচেনা লাগতে পারে।
    CLAHE ব্যবহার করা হচ্ছে কারণ এটা লোকাল কনট্রাস্ট adjust করে, ফলে
    উজ্জ্বল/অন্ধকার আলোতেও face pattern প্রায় একই রকম থাকে।"""
    resized = cv2.resize(gray_face, FACE_SIZE)
    equalized = _clahe.apply(resized)
    return equalized


def train_recognizer():
    """dataset/ ফোল্ডার থেকে ছবি পড়ে LBPH মডেল ট্রেইন করে।"""
    faces, labels = [], []
    label_map = {}
    current_label = 0

    if not os.path.isdir(DATASET_DIR):
        print(f"[!] '{DATASET_DIR}' ফোল্ডার পাওয়া যায়নি। প্রথমে dataset তৈরি করো।")
        return None, {}

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        label_map[current_label] = person_name

        image_count = 0
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"    [!] পড়া যায়নি (করাপ্ট/ভুল ফরম্যাট): {img_name}")
                continue
            detected = face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(80, 80)
            )
            if len(detected) == 0:
                print(f"    [!] মুখ শনাক্ত হয়নি: {img_name}")
                continue
            for (x, y, w, h) in detected:
                face_roi = gray[y:y + h, x:x + w]
                faces.append(preprocess_face(face_roi))
                labels.append(current_label)
                image_count += 1

        print(f"    -> {person_name}: {image_count}টা face sample পাওয়া গেছে")
        if image_count < 15:
            print(f"    [!] সতর্কতা: {person_name}-এর জন্য sample সংখ্যা কম (কমপক্ষে "
                  f"20-30টা ছবি রাখলে recognition ভালো হবে)")

        current_label += 1

    if not faces:
        print("[!] dataset-এ কোনো মুখ পাওয়া যায়নি।")
        return None, {}

    recognizer.train(faces, np.array(labels))
    print(f"[+] {len(label_map)} জনের উপর ট্রেইনিং সম্পন্ন হয়েছে: {list(label_map.values())}")
    return recognizer, label_map


def send_command(url, retries=2, timeout=5):
    """timeout বাড়ানো হয়েছে (2s -> 5s) আর retry যোগ করা হয়েছে, যাতে ESP32
    সাময়িকভাবে ব্যস্ত/রিবুট হলেও কমান্ডটা মিস না হয়ে যায়।"""
    for attempt in range(1, retries + 2):
        try:
            requests.get(url, timeout=timeout)
            print(f"[>] ESP32-কে কমান্ড পাঠানো হলো: {url}")
            return
        except requests.exceptions.RequestException as e:
            print(f"[!] ESP32-এর সাথে সংযোগ করা যায়নি (চেষ্টা {attempt}): {e}")
            if attempt <= retries:
                time.sleep(1.5)


def check_motion():
    try:
        resp = requests.get(ESP32_MOTION_URL, timeout=3)
        return resp.text.strip() == "1"
    except requests.exceptions.RequestException as e:
        print(f"[!] ESP32 ({ESP32_IP}) এর সাথে সংযোগ করা যায়নি: {e}")
        return False


def run_face_recognition_session(recognizer_model, label_map):
    # CAP_DSHOW backend: বারবার open/close করার সময় MSMF backend-এর glitch এড়াতে
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] ওয়েবক্যাম খোলা যাচ্ছে না।")
        return

    print("[i] মোশন ডিটেক্ট হয়েছে - ক্যামেরা চালু হলো, মুখ খোঁজা হচ্ছে...")
    session_start = time.time()
    known_face_found = False
    unknown_face_saved = False
    consecutive_match_name = None
    consecutive_match_count = 0

    while time.time() - session_start < RECOGNITION_WINDOW_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.1, 6, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            processed = preprocess_face(face_roi)
            label, confidence = recognizer_model.predict(processed)
            name_guess = label_map.get(label, "Unknown")

            if DEBUG_MODE:
                print(f"    [debug] predicted={name_guess}  confidence={confidence:.1f}"
                      f"  (threshold={CONFIDENCE_THRESHOLD})")

            if confidence < CONFIDENCE_THRESHOLD:
                name = name_guess
                color = (0, 255, 0)
                text = f"{name} ({confidence:.0f})"

                if name == consecutive_match_name:
                    consecutive_match_count += 1
                else:
                    consecutive_match_name = name
                    consecutive_match_count = 1

                if consecutive_match_count >= REQUIRED_CONSECUTIVE_MATCHES:
                    known_face_found = True
            else:
                name = "Unknown"
                color = (0, 0, 255)
                text = "Unknown"
                consecutive_match_name = None
                consecutive_match_count = 0
                if not unknown_face_saved:
                    filename = os.path.join(
                        UNKNOWN_DIR, f"unknown_{int(time.time())}.jpg"
                    )
                    cv2.imwrite(filename, frame)
                    print(f"[+] অচেনা মুখের ছবি সেভ হলো: {filename}")
                    unknown_face_saved = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Smart Security - Face Recognition (motion triggered)", frame)
        cv2.waitKey(1)

        if known_face_found:
            send_command(ESP32_OPEN_URL)
            time.sleep(1.5)
            break

    if not known_face_found and unknown_face_saved:
        send_command(ESP32_ALERT_URL)

    cap.release()
    cv2.destroyAllWindows()
    print("[i] সেশন শেষ, ক্যামেরা বন্ধ হলো। আবার motion-এর অপেক্ষায়...")


def main():
    print("[i] Dataset থেকে training শুরু হচ্ছে...")
    recognizer_model, label_map = train_recognizer()
    if recognizer_model is None:
        return

    print("[i] সিস্টেম প্রস্তুত। ESP32-এর সেন্সরে motion-এর অপেক্ষায় আছি...")
    print("[i] বন্ধ করতে Ctrl+C চাপো।")

    poll_count = 0
    try:
        while True:
            if check_motion():
                run_face_recognition_session(recognizer_model, label_map)
                time.sleep(COOLDOWN_AFTER_TRIGGER)
            else:
                poll_count += 1
                if poll_count % 10 == 0:
                    print(f"[i] এখনো motion পাওয়া যায়নি, অপেক্ষায় আছি... (poll #{poll_count})")
                time.sleep(MOTION_POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[i] প্রোগ্রাম বন্ধ করা হলো।")


if __name__ == "__main__":
    main()
