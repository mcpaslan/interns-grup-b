import torch
import torchvision
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor

# Önceden eğitilmiş Mask R-CNN modelini yükle (COCO veri seti ile eğitilmiş)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
device = torch.device("cpu")  # CPU üzerinde çalıştır
model.to(device)

# Girdi resmi oku
img_path = "images/forest.jpg"
img_bgr = cv2.imread(img_path)
assert img_bgr is not None, f"Resim Bulunamadı: {img_path}"  # Resim bulunmazsa hata ver
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatında okur → RGB'ye çevir

# Resmi tensöre dönüştür (modelin anlayacağı formata).
tensor = to_tensor(img_rgb).to(device)

# Modelden tahmin al (gradients kapalı → daha hızlı).
with torch.no_grad():
    outputs = model([tensor])

# Çıktıları al
out = outputs[0]
boxes = out["boxes"].numpy()     # Nesne bounding box koordinatları
scores = out["scores"].numpy()   # Nesne tespit doğruluk skorları
labels = out["labels"].numpy()   # Nesne sınıf etiketleri (COCO id)
masks = out["masks"].numpy()     # Nesne maskeleri (piksel bazlı)

# Parametreler
score_tresh = 0.5  # Güven eşiği %50
alpha = 0.5        # Maske saydamlık değeri
overlay = img_bgr.copy()  # Çizim yapılacak kopya resim..

# Her nesne için döngü
for i in range(len(boxes)):
    if scores[i] < score_tresh:  # Güven eşiğinin altındakileri atla
        continue

    # Kutu koordinatlarını al
    x1, y1, x2, y2 = boxes[i].astype(int)

    # Nesne maskesini oluştur (0/1 değerler).
    mask = (masks[i, 0] > 0.5).astype(np.uint8)

    # Rastgele renk seç
    color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()

    # Renkli maske hazırla.
    colored = np.zeros_like(img_bgr, dtype=np.uint8)
    colored[:, :] = color
    mask_3c = np.stack([mask] * 3, axis=-1)  # Maske 3 kanala çıkarılıyor

    # Orijinal resim + renkli maske karıştırılıyor (alpha ile saydamlaştırma).
    overlay = np.where(
        mask_3c == 1,
        cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0),
        overlay
    )

    # Bounding box çiz
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # Etiket + güven skoru yaz
    cv2.putText(
        overlay,
        f"{int(labels[i])} | {scores[i]:.2f}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )

# Çıktıyı kaydet
cv2.imwrite("results/maskrcnn_output.jpg", overlay)
