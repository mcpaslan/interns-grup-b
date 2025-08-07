import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# 1. MoveNet modeli yükleniyor
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# 2. Video dosyasını aç
video_path = "resim/pose1.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Video çıktı ayarları
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("movenet_output.mp4", fourcc, fps, (w, h))

# 4. Keypoint bağlantıları (sadece görsel için)
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (0, 255, 0), (1, 3): (0, 255, 0), (0, 2): (255, 0, 0), (2, 4): (255, 0, 0),
    (0, 5): (0, 255, 255), (0, 6): (255, 255, 0), (5, 7): (0, 255, 255), (7, 9): (0, 255, 255),
    (6, 8): (255, 255, 0), (8, 10): (255, 255, 0), (5, 6): (255, 0, 255),
    (5, 11): (0, 255, 255), (6, 12): (255, 255, 0), (11, 12): (128, 0, 255),
    (11, 13): (0, 255, 255), (13, 15): (0, 255, 255), (12, 14): (255, 255, 0), (14, 16): (255, 255, 0)
}

# 5. Her kare için işlem
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Kareyi model için yeniden boyutlandır
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)

    # Tahmin yap
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    # Her bir nokta için çizim yap
    h_frame, w_frame, _ = frame.shape
    for idx, kp in enumerate(keypoints):
        y, x, confidence = kp
        if confidence > 0.3:
            cx, cy = int(x * w_frame), int(y * h_frame)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Kemikleri çiz
    for (p1, p2), color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > 0.3 and c2 > 0.3:
            x1_int, y1_int = int(x1 * w_frame), int(y1 * h_frame)
            x2_int, y2_int = int(x2 * w_frame), int(y2 * h_frame)
            cv2.line(frame, (x1_int, y1_int), (x2_int, y2_int), color, 2)

    # Göster ve kaydet
    cv2.imshow("MoveNet Pose", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ movenet_output.mp4 oluşturuldu.")
 