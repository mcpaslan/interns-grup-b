import cv2
import pytesseract  # Tesseract OCR motoru için Python arayüzü
import pandas as pd  # Verileri tablo formatında (DataFrame) işlemek için kütüphane
from pytesseract import Output  # Tesseract çıktı formatlarını belirtmek için
import re  # Metin içinde kalıp aramak için (Regular Expressions)


# İsim/soyisim gibi metinsel alanlardaki OCR hatalarını temizlemek için bir fonksiyon.
def clean_text_for_names(text):
    # Fonksiyona gelen verinin metin (string) olup olmadığını kontrol et.
    if not isinstance(text, str):
        return ""

    # Tüm metni büyük harfe çevir.
    text = text.upper()

    # Sık yapılan OCR hatalarını (rakam->harf) düzeltmek için değiştirme listesi.
    replacements = {"0": "O", "1": "I", "5": "S", "8": "B"}
    # Listedeki her bir hatayı metin içinde düzelt.
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Metin içinde sadece izin verilen karakterleri (Türkçe harfler ve boşluk) tut, gerisini sil.
    text = re.sub(r"[^A-ZÇĞİÖŞÜ\s]", "", text)
    # Birden fazla olan boşlukları tek boşluğa indir ve metnin kenarlarındaki boşlukları sil.
    text = re.sub(r"\s+", " ", text).strip()

    # Temizlenmiş metni döndür.
    return text


# --- ANA KOD ---

# İşlenecek resmin dosya yolunu belirt.
image_path = 'images/kimlik.jpg'

# OpenCV ile resmi oku.
img = cv2.imread(image_path)

# Resmin doğru okunup okunmadığını kontrol et.
if img is None:
    print(f"HATA: '{image_path}' dosyası bulunamadı veya okunamadı!")
else:
    print("Görüntü başarıyla yüklendi. Ön işleme başlıyor...")

    # --- GÖRÜNTÜ ÖN İŞLEME ---
    # OCR doğruluğunu artırmak için resmi Tesseract'in seveceği hale getirme adımları.

    # Resmi belirli bir yüksekliğe ölçekle (daha iyi tanıma için).
    target_height = 1200
    scale_ratio = target_height / img.shape[0]
    width = int(img.shape[1] * scale_ratio)
    img_processed = cv2.resize(img, (width, target_height), interpolation=cv2.INTER_LANCZOS4)

    # Resmi renkli formattan gri tonlamaya çevir.
    gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)

    # Arka plan desenleri gibi ince gürültüleri azaltmak için Gaussian Blur filtresi uygula.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Görüntüyü Otsu metoduyla siyah-beyaz yap (Binarization).
    _, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- OCR İŞLEMİ ---

    # Tesseract için yapılandırma ayarları: LSTM motoru, dağınık metin modu, dil Türkçe.
    custom_config = r'--oem 3 --psm 11 -l tur'

    # Ayarlarla birlikte ön işlenmiş resme OCR uygula ve sonuçları bir Pandas DataFrame'e al.
    ocr_data = pytesseract.image_to_data(
        binary_image, config=custom_config, output_type=Output.DATAFRAME
    )

    # --- OCR VERİSİNİ TEMİZLEME ---

    # Güven skoru ('conf') olmayan satırları DataFrame'den kaldır.
    ocr_data.dropna(subset=['conf'], inplace=True)
    # Metin olmayan yapısal bilgi satırlarını (conf = -1) kaldır.
    ocr_data = ocr_data[ocr_data['conf'] != -1]
    # Metinlerin kenar boşluklarını temizle.
    ocr_data['text'] = ocr_data['text'].str.strip()
    # Boş metin olarak okunan satırları kaldır.
    ocr_data = ocr_data[ocr_data['text'] != '']

    print(f"OCR tamamlandı. {len(ocr_data)} adet kelime bulundu.")

    # --- VERİ AYIKLAMA ---

    # 1. Ham Metni Birleştir: DataFrame'deki tüm kelimeleri tek bir metin değişkeninde topla.
    raw_full_text = " ".join(ocr_data['text'].tolist())
    print("\n--- HAM OCR METNİ ---\n", raw_full_text)

    # 2. T.C. Kimlik No'yu Bul:
    # Arama kolaylığı için birleştirilmiş metindeki tüm boşlukları kaldır.
    text_for_tckn = re.sub(r"\s+", "", raw_full_text)
    # RegEx kullanarak 11 haneli rakam grubunu ara.
    tckn_match = re.search(r"(\d{11})", text_for_tckn)

    print("\n--- AYIKLANAN BİLGİLER ---")
    # TCKN bulunduysa ekrana yazdır.
    if tckn_match:
        print(f"T.C. Kimlik No: {tckn_match.group(0)}")
    else:
        print("T.C. Kimlik No bulunamadı.")

    # 3. İsim/Soyisim için Metni Temizle (Opsiyonel):
    # 'clean_text_for_names' fonksiyonunu kullanarak metinleri temizle ve yeni bir sütuna ekle.
    ocr_data['cleaned_text'] = ocr_data['text'].apply(clean_text_for_names)
    # Temizlenmiş kelimeleri de tek bir metin olarak birleştir.
    cleaned_full_text = " ".join(ocr_data['cleaned_text'].tolist())
    print("\n--- SADECE METİN İÇİN TEMİZLENMİŞ VERSİYON ---\n", cleaned_full_text)

    # --- SONUÇLARI GÖRSELLEŞTİRME ---

    # Görüntü üzerine çizilecek kutular için minimum güven skoru eşiği belirle.
    min_confidence = 35
    # Tespit edilen her bir kelime için döngü başlat.
    for i, row in ocr_data.iterrows():
        # Kelimenin güven skorunu ve metnini al.
        conf = int(row['conf'])
        text = row['text']
        # Güven skoru eşikten yüksekse ve metin boş değilse devam et.
        if conf > min_confidence and text:
            # Kelimenin koordinatlarını ve boyutlarını al.
            (x, y, w, h) = (int(row['left']), int(row['top']), int(row['width']), int(row['height']))
            # Bu koordinatları kullanarak resmin üzerine yeşil bir dikdörtgen (bounding box) çiz.
            cv2.rectangle(img_processed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Sonuç görüntüsünü bir pencerede göster.
    cv2.imshow('OCR Sonucu', img_processed)
    # Kullanıcı bir tuşa basana kadar pencereyi açık tut.
    cv2.waitKey(0)
    # Açık olan tüm OpenCV pencerelerini kapat.
    cv2.destroyAllWindows()
    # Not: Kaydetmeden önce 'results' klasörünün var olduğundan emin olun.
    cv2.imwrite('results/kimlik_sonuc.jpg', img_processed)