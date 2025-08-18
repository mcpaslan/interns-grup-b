import cv2
import easyocr
from ultralytics import YOLO
from thefuzz import fuzz
import json

# --- MODELLERİ VE VERİTABANINI BİR KERE YÜKLE (GLOBAL ALAN) ---
try:
    yolo_model = YOLO('model/urun_bul.pt')
    ocr_reader = easyocr.Reader(['tr', 'en'], gpu=False)
    with open('database.json', 'r', encoding='utf-8') as f:
        urun_veritabani = json.load(f)
except Exception as e:
    print(f"Modeller yüklenirken bir hata oluştu: {e}")
    yolo_model = ocr_reader = urun_veritabani = None


# --- ANA FONKSİYON ---
def find_product_details(image_to_process):
    """
    Bu fonksiyon bir CV2 formatında resim alır, içindeki ürünleri bulur,
    bilgilerini veritabanından çeker ve sonuçları döndürür.
    """
    if yolo_model is None:
        return image_to_process, []

    image_copy = image_to_process.copy()
    results = yolo_model.predict(source=image_copy, verbose=False)  # 'verbose=False' terminali temiz tutar
    detected_products_info = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cropped_product = image_copy[y1:y2, x1:x2]

            if cropped_product.size == 0:
                continue

            gray_image = cv2.cvtColor(cropped_product, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # --- OCR PERFORMANSINI ARTIRMAK İÇİN PROAKTİF ÇÖZÜM ---
            # Eğer kırpılan görüntü çok küçükse, OCR için büyütülüyor.
            height, width = binary_image.shape
            if height > 0 and width > 0 and (height < 50 or width < 50):  # Yükseklik veya genişlik 50 pikselden küçükse
                # Oranı koruyarak yüksekliği 100 piksel yapacak şekilde büyüt
                scale_factor = 100 / height
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                print(
                    f"[DEBUG] Görüntü çok küçük, büyütülüyor: ({width}, {height}) -> ({new_width}, {new_height})")  # <-- DEBUG KODU
                binary_image = cv2.resize(binary_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            ocr_results = ocr_reader.readtext(binary_image)

            en_yuksek_benzerlik = 0
            tespit_edilen_urun_key = None

            for (bbox, text, prob) in ocr_results:
                for urun_key, urun_bilgisi in urun_veritabani.items():
                    for anahtar_kelime in urun_bilgisi.get('anahtar_kelimeler', []):
                        benzerlik_skoru = fuzz.ratio(anahtar_kelime.lower(), text.lower())
                        if benzerlik_skoru > en_yuksek_benzerlik and benzerlik_skoru > 75:
                            en_yuksek_benzerlik = benzerlik_skoru
                            tespit_edilen_urun_key = urun_key

            if tespit_edilen_urun_key:
                detected_products_info.append(urun_veritabani[tespit_edilen_urun_key])

    return image_copy, detected_products_info

# --- 3. TEST ALANI ---
if __name__ == '__main__':
    print("--- urun_icerik_bul.py Test Modunda Çalıştırılıyor ---")
    test_image_path = 'pictures/kola.jpg'  # Test için kendi resim yolunu kullanabilirsin
    test_image = cv2.imread(test_image_path)

    if test_image is not None:
        processed_image, found_products = find_product_details(test_image)

        if found_products:
            print(f"\n{len(found_products)} adet ürün bulundu:")
            for product in found_products:
                print(f"  - Tam İsim: {product['tam_isim']}")
        else:
            print("\nTest resminde veritabanıyla eşleşen bir ürün bulunamadı.")
    else:
        print(f"HATA: Test resmi '{test_image_path}' bulunamadı veya okunamadı.")