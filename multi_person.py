import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

# -----------------------
# Hız / Doğruluk Ayarları
# -----------------------
YOLO_WEIGHTS = "yolov8n.pt"
YOLO_IMGSZ   = 512
POSE_STRIDE  = 3
ROI_SCALE    = 0.5

# Squat eşikleri (diz açısı)
DOWN_THRESH = 90
UP_THRESH   = 160

# -----------------------
# Pose yardımcıları
# -----------------------
mp_pose = mp.solutions.pose
pose_est = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def angle(a, b, c):
    a, b, c = map(lambda x: np.array(x, dtype=np.float32), (a, b, c))
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def classify_from_roi(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    h0, w0 = roi_bgr.shape[:2]
    if h0 < 20 or w0 < 20:
        return None

    if ROI_SCALE != 1.0:
        roi_bgr = cv2.resize(roi_bgr, None, fx=ROI_SCALE, fy=ROI_SCALE, interpolation=cv2.INTER_LINEAR)

    h, w = roi_bgr.shape[:2]
    res = pose_est.process(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    p  = lambda i: (int(lm[i].x * w), int(lm[i].y * h))
    lh, lk, la = p(mp_pose.PoseLandmark.LEFT_HIP), p(mp_pose.PoseLandmark.LEFT_KNEE), p(mp_pose.PoseLandmark.LEFT_ANKLE)
    rk, ra     = p(mp_pose.PoseLandmark.RIGHT_KNEE), p(mp_pose.PoseLandmark.RIGHT_ANKLE)

    left_ang  = angle(lh, lk, la)
    right_ang = angle(lh, rk, ra)  # not: sağ kalça yerine sol kalça referansı
    return (left_ang + right_ang) / 2.0

# -----------------------
# Tespit + Takip
# -----------------------
model   = YOLO(YOLO_WEIGHTS)
tracker = DeepSort(max_age=30, embedder=None)  # embedder kapalı

# id -> durum/süre tutucular
per_id_state       = {}   # 0=Standing, 1=Squatting
per_id_pose        = {}
per_id_start_frame = {}

# -----------------------
# Video Döngüsü
# -----------------------
cap = cv2.VideoCapture("/Users/yunusemreozturk/Desktop/squat4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1e-3:
    fps = 30.0

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]


    yres = model(frame, imgsz=YOLO_IMGSZ, conf=0.45, iou=0.5, classes=[0], verbose=False)[0]
    detections = []
    for b in yres.boxes:
        if int(b.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        w = x2 - x1; h = y2 - y1
        detections.append(([x1, y1, w, h], conf, "person"))


    if detections:
        embeds = [np.ones(128, dtype=np.float32) for _ in detections]  # güvenli sahte vektör
        tracks = tracker.update_tracks(detections, embeds=embeds)
    else:
        tracks = tracker.update_tracks([])


    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        # ID için ilk kurulum
        if tid not in per_id_state:
            per_id_state[tid]       = 0
            per_id_pose[tid]        = "Standing"
            per_id_start_frame[tid] = frame_idx

        # POSE_STRIDE: her N karede bir pose bak
        if frame_idx % POSE_STRIDE == 0:
            roi = frame[y1:y2, x1:x2]
            avg_angle = classify_from_roi(roi)
            if avg_angle is not None:
                new_pose = per_id_pose[tid]
                if per_id_state[tid] == 0 and avg_angle < DOWN_THRESH:
                    new_pose = "Squatting"
                    per_id_state[tid] = 1
                    if new_pose != per_id_pose[tid]:
                        per_id_pose[tid] = new_pose
                        per_id_start_frame[tid] = frame_idx
                elif per_id_state[tid] == 1 and avg_angle > UP_THRESH:
                    new_pose = "Standing"
                    per_id_state[tid] = 0
                    if new_pose != per_id_pose[tid]:
                        per_id_pose[tid] = new_pose
                        per_id_start_frame[tid] = frame_idx

        # Süre (video zamanına göre)
        dwell_s = (frame_idx - per_id_start_frame[tid]) / fps

        # Çizimler
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(12, y1 - 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{per_id_pose[tid]} ({dwell_s:.1f}s)", (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_idx += 1

    cv2.imshow("Per-ID Pose + Dwell (video time)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
