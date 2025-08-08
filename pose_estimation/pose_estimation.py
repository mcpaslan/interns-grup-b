import cv2
import mediapipe as mp

# MediaPipe pose ve çizim yardımcıları
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Statik görüntü için poz tahminleyici
pose = mp_pose.Pose(
    static_image_mode=True,       # fotoğraf modunda çalış
    model_complexity=1,           # 0–2: doğruluk/hız dengesi
    enable_segmentation=False,    # segmentasyon kapalı
    min_detection_confidence=0.5  # tespit eşiği
)

# Görseli yükle
image_path = "basketball.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

h, w = image.shape[:2]

# BGR → RGB (MediaPipe RGB ister)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Poz tahmini
results = pose.process(img_rgb)

# Landmark varsa çiz ve etiketle
if results.pose_landmarks:
    # İskelet çizimi
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
    )

    # Etiketlenecek eklemler
    landmark_names = {
        "Head (Nose)": mp_pose.PoseLandmark.NOSE,
        "Left Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "Right Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "Left Elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "Right Elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "Left Wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "Right Wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "Left Hip": mp_pose.PoseLandmark.LEFT_HIP,
        "Right Hip": mp_pose.PoseLandmark.RIGHT_HIP,
        "Left Knee": mp_pose.PoseLandmark.LEFT_KNEE,
        "Right Knee": mp_pose.PoseLandmark.RIGHT_KNEE,
        "Left Ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
        "Right Ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    }

    print("Detected keypoint coordinates (pixels):")
    for name, lm_enum in landmark_names.items():
        lm = results.pose_landmarks.landmark[lm_enum]
        x_px = int(lm.x * w)  # [0–1] → piksel
        y_px = int(lm.y * h)
        visibility = lm.visibility
        print(f"{name}: ({x_px}, {y_px}) visibility={visibility:.2f}")
        cv2.putText(image, name, (x_px + 5, y_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)
        cv2.circle(image, (x_px, y_px), 5, (255,0,0), -1)
else:
    print("No person/pose detected.")

# Göster
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
