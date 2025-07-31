import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO("models/futbol.pt")

# Görseli yükle
image_path = "images/football_players.jpg"
image = cv2.imread(image_path)

# Tespit yap
results = model(image)

# İnsan sınıfını filtrele (class id == 0)
classes = results[0].boxes.cls
person_count = (classes == 0).sum().item()

# Görsele sonuçları çiz
annotated_image = results[0].plot()

# İnsan sayısını görselin üzerine yaz
cv2.putText(
    annotated_image,
    f"People detected: {person_count}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# Görseli göster
cv2.imshow("Detected People", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (Opsiyonel) Görseli kaydet
cv2.imwrite("results/output_with_count.jpg", annotated_image)
