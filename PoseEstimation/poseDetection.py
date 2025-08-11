import cv2
import numpy as np
from ultralytics import YOLO
import collections
import time

# Model yükle
model = YOLO("models/yolov8n-pose.pt")  # Daha iyi doğruluk için medium model

# Video yükle
cap = cv2.VideoCapture("videos/mixed.mp4")  # Burada istediğin videoyu kullan

# Çıktı videosu ayarları
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("results/pose_all_in_one.mp4", fourcc, fps, (width, height))

# Parametreler
CONF_THRESHOLD = 0.5
STEP_RATIO = 0.25  # Walking için adım-genişlik oranı eşiği
ANGLE_SIT_THRESHOLD = 110  # Diz açısı oturma eşiği

# Kişi durumu: {id: {"last_status": str, "start_time": float}}
person_states = {}

# Açı hesaplama fonksiyonu
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

person_states = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()
    keypoints_data = results[0].keypoints.data
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for i, keypoints in enumerate(keypoints_data):
        if keypoints.shape[0] == 0:
            continue

        l_hip, r_hip = 11, 12
        l_knee, r_knee = 13, 14
        l_ankle, r_ankle = 15, 16

        conf = results[0].keypoints.conf[i].cpu().numpy()
        if np.any(conf[[l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]] < CONF_THRESHOLD):
            continue

        left_hip = keypoints[l_hip][:2]
        right_hip = keypoints[r_hip][:2]
        left_knee = keypoints[l_knee][:2]
        right_knee = keypoints[r_knee][:2]
        left_ankle = keypoints[l_ankle][:2]
        right_ankle = keypoints[r_ankle][:2]

        left_angle = calculate_angle(left_hip, left_knee, left_ankle)

        step_width = abs(left_ankle[0] - right_ankle[0])
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        torso_height = abs(hip_y - ankle_y)
        walking_detected = torso_height > 0 and step_width > STEP_RATIO * torso_height

        if left_angle < ANGLE_SIT_THRESHOLD:
            status = "Sitting"
        elif walking_detected:
            status = "Walking"
        else:
            status = "Standing"

        # Kişi takibi
        if i not in person_states:
            person_states[i] = {"last_status": status, "frame_count": 0}

        if person_states[i]["last_status"] != status:
            person_states[i]["last_status"] = status
            person_states[i]["frame_count"] = 0  # yeni durum için sıfırla
        else:
            person_states[i]["frame_count"] += 1

        # FPS'e göre süre hesaplama
        elapsed = person_states[i]["frame_count"] / fps

        x1, y1, x2, y2 = boxes[i]
        label = f"{status} ({elapsed:.1f}s)"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x2 - 200, y2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("Pose Detection", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
