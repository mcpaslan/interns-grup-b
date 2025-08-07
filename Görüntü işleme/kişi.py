from ultralytics import YOLO
import cv2

# 1. Modeli yükle
model = YOLO("yolov8n.pt")

# 2. Video ve kayıt ayarları
video_path = "resim/insanlar.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("cikti_yolo.mp4", fourcc, fps, (width, height))
cap.release()

# 3. Kişi takip sistemi için veri yapıları
counted_ids = set()         # Tekil gerçek ID'leri tutar
custom_id_map = {}          # {gerçek track_id: bizim sıralı ID}
next_custom_id = 0          # Bir sonraki verilecek sıralı ID

# 4. Her kareye kutu ve yazı ekleyen fonksiyon
def annotate(frame, results):
    global next_custom_id  # fonksiyon içinde artırmak için

    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # sadece insan sınıfı
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is not None:
                # Daha önce bu track_id'ye özel ID verilmemişse ata
                if track_id not in custom_id_map:
                    custom_id_map[track_id] = next_custom_id
                    next_custom_id += 1

                display_id = custom_id_map[track_id]
                counted_ids.add(track_id)

                # Kutu çizimi
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ID yazısı
                cv2.putText(frame, f"Kisi {display_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sol üst köşeye toplam kişi sayısını yaz
    cv2.putText(frame, f"Toplam Kisi: {len(counted_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# 5. YOLO ile video takibi
results = model.track(
    source=video_path,
    tracker="botsort.yaml",
    persist=True,
    classes=[0],    # sadece insan
    stream=True
)

# 6. Her kareyi işle ve kaydet
for r in results:
    frame = r.orig_img.copy()
    frame = annotate(frame, [r])
    out.write(frame)

# 7. Video yazıcıyı kapat
out.release()
print("✅ cikti_yolo.mp4 oluşturuldu.")
