import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
import time
import csv
import json

model = YOLO("models/yolov8n-pose.pt")

cap = cv2.VideoCapture("videos/sitting.mp4")

# Çıktı için codec'i tanımladık  ve bir VideoWriter nesnesi oluşturduk.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_width = 1080
output_height = 1920
person_states = {}  # {person_id: {"last_status": ..., "start_time": ..., "history": [...]}}
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('results/sitting_pose.mp4', fourcc, fps, (output_width, output_height))

# Üç anahtar nokta arasındaki açıyı hesaplamak için bir fonksiyon tanımla
def calculate_angle(a, b, c):
    a = np.array(a)  # İlk
    b = np.array(b)  # Orta
    c = np.array(c)  # Son

    # Radyan cinsinden açıyı hesaplar ve dereceye çevirir.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    # Açı 180 dereceden büyükse, 360'dan çıkararak düzeltir.

    if angle > 180.0:
        angle = 360 - angle

    return angle


while True:
    ret, frame = cap.read()

    # Frame'i yeniden boyutlandır, başarısız olursa geç
    try:
        frame = cv2.resize(frame, (output_width, output_height))
    except:
        pass

    # Daha fazla frame kalmadığında döngüyü kır
    if not ret:
        break

    # model ile tahmin yap
    results = model.predict(frame, save=False)
    annotated = results[0].plot()  # iskelet çizimi
    # Sınırlayıcı kutu bilgilerini xyxy formatında al
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    statuses = []

    # Algılanan tüm kişiler için anahtar nokta verilerini al
    keypoints_data = results[0].keypoints.data

    # Algılanan kişileri yinele
    for i, keypoints in enumerate(keypoints_data):
        if keypoints.shape[0] > 0:
            # 1. Açı hesapla
            angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
            # 2. Oturma durumu belirle
            status = 'Sitting' if angle is not None and angle < 110 else 'Standing'
            statuses.append(status)

            # 3. Kişi durumu kayıtlı değilse oluştur
            if i not in person_states:
                person_states[i] = {
                    "last_status": status,
                    "start_time": frame_count,
                    "history": []
                }

            # 4. Durum değişmişse geçmişe kaydet
            if person_states[i]["last_status"] != status:
                start = person_states[i]["start_time"]
                end = frame_count
                duration_seconds = (end - start) / fps

                person_states[i]["history"].append({
                    "status": person_states[i]["last_status"],
                    "start_frame": start,
                    "end_frame": end,
                    "duration_sec": round(duration_seconds, 2)
                })

                person_states[i]["last_status"] = status
                person_states[i]["start_time"] = frame_count

            # 5. Açı yazdır
            print(f"Kişi {i + 1} {status} (Açı: {angle:.2f} derece)")
    # Frame üzerine sınırlayıcı kutuları ve durumları çiz
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(
            annotated, f"{statuses[i]}", (x1, y2 - 10),
            scale=3, thickness=3,
            colorT=(255, 255, 255), colorR=(255, 0, 255),
            font=cv2.FONT_HERSHEY_PLAIN,
            offset=10,
            border=0, colorB=(0, 255, 0)
        )

    # Frame'i çıktı video dosyasına yaz
    out.write(annotated)

    cv2.imshow('YOLOv8 Sitting Position', annotated)
    if cv2.waitKey(1) == ord('q'):
        break
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Kalan son durumu da history'ye ekle (tam çıkışta)
for i in person_states:
    start = person_states[i]["start_time"]
    end = frame_count
    duration_seconds = (end - start) / fps
    person_states[i]["history"].append({
        "status": person_states[i]["last_status"],
        "start_frame": start,
        "end_frame": end,
        "duration_sec": round(duration_seconds, 2)
    })

# JSON olarak kaydet
with open("results/behavior_log.json", "w") as json_file:
    json.dump(person_states, json_file, indent=4)

# CSV olarak kaydet (isteğe bağlı)
with open("results/behavior_log.csv", "w", newline='') as csvfile:
    fieldnames = ["person_id", "status", "start_frame", "end_frame", "duration_sec"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for pid, data in person_states.items():
        for record in data["history"]:
            writer.writerow({
                "person_id": pid,
                **record
            })