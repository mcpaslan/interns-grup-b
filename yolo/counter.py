#bitmedi, hatalı çalışıyor. düzeltilecek
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

unique_ids = set()
stream = model.track(
    source='people.mp4',
    tracker='bytetrack.yaml',
    stream=True,
    device='cpu',
    imgsz=640,
    conf=0.3,
    iou=0.5
)

for r in stream:
    frame = r.orig_img

    for box in r.boxes:
        if int(box.cls[0]) != 0:  # sadece person
            continue
        tid = int(box.id[0])
        unique_ids.add(tid)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    active = sum(1 for box in r.boxes if int(box.cls[0]) == 0)
    cv2.putText(frame,
                f"Active: {active}  Total unique: {len(unique_ids)}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Tracked People", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
