import cv2
from ultralytics import YOLO
import collections

model = YOLO('models/yolov8m-pose.pt')  # Daha iyi sonuçlar için medium model kullanıldı
cap = cv2.VideoCapture("videos/running.mp4")  # Koşma videosu kullanılmalı

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("results/running_pose.mp4", fourcc, fps, (width, height))

# Parametreler
CONF_THRESHOLD = 0.5  # Minimum güven skoru
FLIGHT_THRESHOLD = 5  # Ayakların yerden kesilme eşiği
y_history_window = 10  # Kaç karelik geçmiş tutulacak
# Her kişi için geçmiş ankle-y değerlerini tutan yapı
y_histories = collections.defaultdict(lambda: collections.deque(maxlen=y_history_window))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False) # Modelden tahmin al
    annotated = results[0].plot()

    # Her bir kişi için döngü
    for idx, person in enumerate(results[0].keypoints):
        if person is None or person.xy is None or person.conf is None:
            continue

        points = person.xy[0]        # [17, 2]
        confs = person.conf[0]       # [17]

        try:
            l_ankle, r_ankle = 15, 16

            if confs[l_ankle] < CONF_THRESHOLD or confs[r_ankle] < CONF_THRESHOLD:
                continue
            # Sol ve sağ bileğin Y koordinatları alınır
            left_ankle_y = points[l_ankle][1]
            right_ankle_y = points[r_ankle][1]
            current_y = (left_ankle_y + right_ankle_y) / 2  # Ortalama y değeri

            # Bu kişiye özel geçmiş y değerlerini güncelle
            y_history = y_histories[idx]
            y_history.append(current_y)

            is_running = False
            # Geçmiş dolduğunda koşma tespiti yapılabilir
            if len(y_history) == y_history.maxlen:
                min_y = min(y_history) # Son karelerdeki en düşük y (en yukarıdaki ayak pozisyonu)
                # Eğer anlık ayak yüksekliği bu değerden belirgin şekilde yukarıdaysa kişi zıplamaktadır
                if (left_ankle_y < min_y - FLIGHT_THRESHOLD) or (right_ankle_y < min_y - FLIGHT_THRESHOLD):
                    is_running = True

            if is_running:
                # Etiket yazılacak konum: ayak bileklerinin ortası
                x = int((points[l_ankle][0] + points[r_ankle][0]) / 2)
                y = int((points[l_ankle][1] + points[r_ankle][1]) / 2)
                cv2.putText(annotated, "Running", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

        except Exception:
            continue

    cv2.imshow("YOLOv8 Running Detection", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
