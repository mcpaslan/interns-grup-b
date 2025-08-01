

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
video_path = "insanlar.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("cikti_yolo.mp4", fourcc, fps, (width, height))
cap.release()

counted_ids = set()

def annotate(frame, results):
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is not None:
                counted_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Kisi {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Toplam Kisi: {len(counted_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

results = model.track(
    source=video_path,
    tracker="botsort.yaml",
    persist=True,
    classes=[0],
    stream=True
)

for r in results:
    frame = r.orig_img.copy()
    frame = annotate(frame, [r])
    out.write(frame)

out.release()
print("✅ cikti_yolo.mp4 oluşturuldu.")
