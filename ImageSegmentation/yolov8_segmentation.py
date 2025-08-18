import cv2
import numpy as np
from ultralytics import YOLO

image_path = r"images/football_players.jpg"
model_path = "yolov8n-seg.pt"
save_path = "results/output.jpg"
conf_threshold = 0.5

# Modeli yükle
model = YOLO(model_path)

# Görüntüyü oku
image = cv2.imread(image_path)

# Tahmin yap.
result = model.predict(source=image, conf=conf_threshold)[0]

# Mask bilgisi al.
masks = result.masks.data.cpu().numpy() if result.masks is not None else []

# Eğer mask yoksa boş bir maske oluştur
if len(masks) == 0:
    semantic_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
else:
    semantic_mask = np.zeros_like(masks[0], dtype=bool)
    for i in range(len(masks)):
        semantic_mask |= masks[i].astype(bool)

# Maskeyi 3 kanallı hale getir.
semantic_mask_uint8 = semantic_mask.astype(np.uint8) * 255
semantic_colored = np.stack([semantic_mask_uint8] * 3, axis=-1)

# Instance segmentation görüntüsü
instance_image = result.plot()

# Boyutları eşitle
size = (640, 640)
image_resized = cv2.resize(image, size)
instance_resized = cv2.resize(instance_image, size)
semantic_resized = cv2.resize(semantic_colored, size)

# Başlık ekleme fonksiyonu
def add_title(img, title):
    cv2.putText(img, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return img

# Başlık ekle
image_resized = add_title(image_resized, "Original")
instance_resized = add_title(instance_resized, "Instance Segmentation")
semantic_resized = add_title(semantic_resized, "Semantic Segmentation")

# Yan yana birleştir
combined = np.hstack((image_resized, instance_resized, semantic_resized))

# Kaydet
cv2.imwrite(save_path, combined)
