import cv2
import easyocr
from ultralytics import YOLO
from thefuzz import fuzz
import json

# --- MODELLERİ VE VERİTABANINI BİR KERE YÜKLE (GLOBAL ALAN) ---
try:
    yolo_model = YOLO('model/urun_bul_v2.pt')
    ocr_reader = easyocr.Reader(['tr', 'en'], gpu=False)
    with open('database.json', 'r', encoding='utf-8') as f:
        urun_veritabani = json.load(f)
except Exception as e:
    print(f"Modeller yüklenirken bir hata oluştu: {e}")
    yolo_model = ocr_reader = urun_veritabani = None


# --- ANA FONKSİYON (GÜNCELLENDİ) ---
def find_product_details(image_to_process):
    if yolo_model is None:
        return image_to_process, []

    # Çizimler için orijinal görüntünün bir kopyasını oluşturuyoruz.
    output_image = image_to_process.copy()
    results = yolo_model.predict(source=image_to_process, verbose=False)

    unique_detected_keys = set()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            # Dikdörtgeni, üzerinde başka çizimler olmayan kopya görüntüye çiziyoruz.
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 1. DÜZELTME: Kırpma işlemini her zaman orijinal, temiz görüntüden yapıyoruz.
            # Bu, bir önceki kutunun çiziminin sonrakini etkilemesini önler.
            cropped_product = image_to_process[y1:y2, x1:x2]

            if cropped_product.size == 0:
                continue

            gray_image = cv2.cvtColor(cropped_product, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            height, width = binary_image.shape
            if height > 0 and width > 0 and (height < 50 or width < 50):
                scale_factor = 100 / height
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                binary_image = cv2.resize(binary_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            ocr_results = ocr_reader.readtext(binary_image)

            # 2. DÜZELTME: Eşleştirme mantığını daha sağlam hale getiriyoruz.
            # Bu kutu içinde bulunan her ürün adayı için en yüksek skoru saklayacağız.
            box_matches = {}

            for (bbox, text, prob) in ocr_results:
                for urun_key, urun_bilgisi in urun_veritabani.items():
                    for anahtar_kelime in urun_bilgisi.get('anahtar_kelimeler', []):
                        benzerlik_skoru = fuzz.ratio(anahtar_kelime.lower(), text.lower())

                        # Eşik değerini geçen her potansiyel eşleşmeyi dikkate alıyoruz.
                        if benzerlik_skoru > 70:  # Eşik değeri daha esnek olması için 70'e çekildi.
                            # Eğer bu ürün için daha yüksek bir skor bulursak güncelliyoruz.
                            if benzerlik_skoru > box_matches.get(urun_key, 0):
                                box_matches[urun_key] = benzerlik_skoru

            # Eğer bu kutu için herhangi bir eşleşme bulunduysa, en yüksek skorlu olanı seçiyoruz.
            if box_matches:
                tespit_edilen_urun_key = max(box_matches, key=box_matches.get)
                unique_detected_keys.add(tespit_edilen_urun_key)

    final_products_info = []
    for key in unique_detected_keys:
        final_products_info.append(urun_veritabani[key])

    return output_image, final_products_info


# --- TEST ALANI ---
if __name__ == '__main__':
    # ... (Test alanı aynı kalabilir) ...
    pass
