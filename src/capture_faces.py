"""
Dataset Capture Tool (Fixed - সঠিক ক্যামেরা বাছাইয়ের সুবিধাসহ)
---------------------------------------------------------------
এই স্ক্রিপ্ট চালিয়ে webcam দিয়ে কোনো ব্যক্তির মুখের ছবি তুলে
dataset/<person_name>/ ফোল্ডারে সেভ করা হয়। এই ছবিগুলো
face_recognition_client.py ট্রেইনিং-এর জন্য ব্যবহার করবে।

⚠️ গুরুত্বপূর্ণ: এখানে যে CAMERA_INDEX ব্যবহার করবে, ঠিক সেই একই
CAMERA_INDEX মূল recognition script-এও ব্যবহার করো। দুই জায়গায় আলাদা
ক্যামেরা ব্যবহার করলে lighting/quality আলাদা হয়ে recognition খারাপ হতে পারে।

চালানোর নিয়ম:
    python capture_faces.py

এরপর নাম লিখবে, তারপর SPACE চাপলে ছবি উঠবে, ESC চাপলে বন্ধ হবে।
একজনের জন্য কমপক্ষে ২৫-৩০টা ছবি তোলা ভালো (বিভিন্ন angle, expression, lighting)।
"""

import cv2
import os

DATASET_DIR = "../dataset"
IMAGES_PER_PERSON_SUGGESTED = 25

# ল্যাপটপের বিল্ট-ইন ক্যামেরা সাধারণত 0, external webcam সাধারণত 1 বা তার বেশি।
# find_camera_index.py চালিয়ে সঠিক নাম্বারটা বের করে এখানে বসাও।
# মূল recognition script-এর CAMERA_INDEX-এর সাথে এটা মিলিয়ে রাখো।
CAMERA_INDEX = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def main():
    person_name = input("যার ছবি তোলা হবে তার নাম লেখো (স্পেস ছাড়া, যেমন: abir): ").strip()
    if not person_name:
        print("[!] নাম ছাড়া কাজ করা যাবে না।")
        return

    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[!] ক্যামেরা index {CAMERA_INDEX} খোলা যাচ্ছে না।")
        print("[!] find_camera_index.py চালিয়ে সঠিক index বের করো এবং "
              "এই ফাইলের উপরে CAMERA_INDEX বদলাও।")
        return

    count = len(os.listdir(person_dir))
    print(f"[i] ক্যামেরা index {CAMERA_INDEX} ব্যবহার করা হচ্ছে।")
    print(f"[i] '{person_name}' এর জন্য ছবি তোলা শুরু।")
    print("[i] SPACE চাপলে ছবি উঠবে, ESC চাপলে বন্ধ হবে।")
    print(f"[i] কমপক্ষে {IMAGES_PER_PERSON_SUGGESTED}টা ছবি তোলার চেষ্টা করো "
          f"(মাথা ঘুরিয়ে, একটু কাছে-দূরে গিয়ে, আলো-অন্ধকারে)।")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] ফ্রেম পড়া যাচ্ছে না।")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display_frame, f"Saved: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Dataset Capture - SPACE=capture, ESC=quit", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("[!] কোনো মুখ পাওয়া যায়নি, আবার চেষ্টা করো।")
                continue
            filename = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"[+] সেভ হলো: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[i] মোট {count}টা ছবি সেভ হয়েছে '{person_dir}' ফোল্ডারে।")


if __name__ == "__main__":
    main()