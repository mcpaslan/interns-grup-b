import cv2
from ultralytics import YOLO

# Modeli yükle (COCO ön-eğitimli, "person" sınıfı ID=0)
model = YOLO('yolov8n.pt')

# Genel fonksiyon: bir frame üzerinde tespit yap ve çiz
def detect_and_draw(frame, conf_thres=0.3):
    results = model(frame)[0]  # tek görüntü için ilk sonuç
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0: continue             # sadece person
        conf = float(box.conf[0])
        if conf < conf_thres: continue    # eşiğin altındakileri atla
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

# --- Tek bir resim üzerinde ---
img = cv2.imread('insan.jpg')
out = detect_and_draw(img.copy(), conf_thres=0.5)
cv2.imshow('Person Detection', out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Webcam veya video dosyası üzerinden ---
"""cap = cv2.VideoCapture("video.mp4")      # 
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = detect_and_draw(frame, conf_thres=0.3)
    cv2.imshow('YOLO Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()"""
cv2.destroyAllWindows()
