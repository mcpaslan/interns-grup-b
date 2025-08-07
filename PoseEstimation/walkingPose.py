from ultralytics import YOLO
import cv2


model = YOLO('models/yolov8m-pose.pt')  # Daha iyi sonuçlar için medium modeli kullanıldı.
cap = cv2.VideoCapture("videos/walking.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("results/walking_pose.mp4", fourcc, fps, (width, height))

# Parametreler
CONF_THRESHOLD = 0.5 #  Keypointler için minimum güven değeri
STEP_RATIO = 0.25 # Adım genişliğinin, gövde yüksekliğine oranıdır.
# Kişinin yürüme durumunu bacak pozisyonuna göre belirlemek için kullanılır.


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot() #iskelet çizimi

    # Her bir tespit edilen insan için:
    for person in results[0].keypoints: #tespit edilen kişilerin bir listesidir —
                    # her biri baş, omuzlar, kalçalar gibi eklem noktalarına sahiptir.
        if person is None or person.xy is None or person.conf is None:
            continue
        #Eğer model boş veya geçersiz bir tespit döndürürse veya güven skoru yoksa
                    # bu kişi atlanır ve sonraki kişiye geçer.
        points = person.xy[0]        # [17, 2]
        confs = person.conf[0]       # [17]

        try:
            l_hip, r_hip = 11, 12
            l_ankle, r_ankle = 15, 16

            if confs[l_hip] < CONF_THRESHOLD or confs[r_hip] < CONF_THRESHOLD:
                continue
            if confs[l_ankle] < CONF_THRESHOLD or confs[r_ankle] < CONF_THRESHOLD:
                continue
            #Bölgelerdeki eşik değerler, default olarak verilenden(%50)düşükse bir sonraki kişiye geçer.

            #Kişinin sağ, sol; ayak ve kalça bölgesinin x ve y konumlarını alır.
            left_hip = points[l_hip]
            right_hip = points[r_hip]
            left_ankle = points[l_ankle]
            right_ankle = points[r_ankle]

            #Bu değerler, bacak açıklığına göre yürüyüp yürümediğine karar verir
            step_width = abs(left_ankle[0] - right_ankle[0]) #Adım genişliği
            hip_y = (left_hip[1] + right_hip[1]) / 2 #Kalça için dikey pozisyon
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2 #Bilekler için dikey pozisyon
            torso_height = abs(hip_y - ankle_y) #Gövde yüksekliğini hesaplar

            # Adım genişliğini kontrol eder. Yürüyüp yürümediği burada belirtilir.
            if torso_height > 0 and step_width > STEP_RATIO * torso_height:
                x = int((left_ankle[0] + right_ankle[0]) / 2)
                y = int((left_ankle[1] + right_ankle[1]) / 2)
                #yukarıdaki kodlar, ayak bilekleri arasındaki açıyı bulur ve alttaki kodla etiketler.
                cv2.putText(annotated, "Walking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

        except Exception as e:
            continue

    cv2.imshow("YOLOv8 Walking Position", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
