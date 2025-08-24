import pytesseract
from PIL import Image
import cv2
import re
from datetime import datetime


def preprocess_image(img_path):
    """
    Verilen yoldaki görüntüyü okur ve OCR işlemi için ön hazırlıktan geçirir.
    Bu fonksiyon, görüntüyü yeniden boyutlandırır, gri tonlamaya çevirir,
    gürültüyü azaltır ve metnin daha belirgin hale gelmesi için ikili (siyah-beyaz)
    formata dönüştürür.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Görüntüyü, orijinal en-boy oranını koruyarak 2 kat büyüt.
    # Bu işlem, Tesseract'in küçük karakterleri daha iyi tanımasına yardımcı olur.
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltırken kenarları korumak için bilateral filtre
    blur = cv2.bilateralFilter(gray, 9, 90, 30)

    # Adaptif thresholding ile görüntüyü siyah-beyaz (ikili) formata dönüştür.
    binary_image = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
    )

    # görüntüdeki küçük noktaları ve gürültüleri temizle.
    # Bu işlem, harflerin bütünlüğünü bozmadan istenmeyen küçük lekeleri yok eder.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # işlenmiş görüntüyü test sırasında görmek için kullanılıyor.
    """cv2.imshow("İşlenmiş Görüntü", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return binary_image


def extract_data_with_logic(text):
    """
    OCR ile metne çevrilmiş fatura içeriğinden, gelişmiş kurallar ve yedek
    mekanizmalar kullanarak yapılandırılmış verileri (Tür, Tutar, Tarih) çıkarır.
    """
    data = {
        "Tür": "Bilinmiyor",
        "Ödenecek Tutar": "Bulunamadı",
        "Son Ödeme Tarihi": "Bulunamadı",
        "Geçmiş Dönem Borcu": "Bulunamadı"
    }
    # Büyük/küçük harf duyarlılığını ortadan kaldırmak için tüm metni küçük harfe çevirir.
    lower_text = text.lower()

    # ----- FATURA TÜRÜ BELİRLENİYOR (ELEKTRİK, SU, DOGALGAZ) ------
    # En spesifik anahtar kelimelerden en genele doğru kontrol ediyoruz.
    if "elektrik" in lower_text or "enerjisa" in lower_text or "bedaş" in lower_text or "boğaziçi" in lower_text or "azici" in lower_text:
        data["Tür"] = "Elektrik Faturası"
    elif "igdaş" in lower_text or "doğalgaz" in lower_text or "gaz" in lower_text or "doğal" in lower_text or "dogal" in lower_text or "GDA" in lower_text:
        # "gaz" kelimesi çok genel olduğu ve yanlış eşleşmelere yol açtığı için kaldırıldı.
        data["Tür"] = "Doğalgaz Faturası"
    elif "su fatura" in lower_text or "iski" in lower_text or "su ve kanalizasyon" in lower_text or "İSİ" in lower_text or "su" in lower_text:
        # "su" kelimesi çok genel olduğu için tek başına kontrol edilmiyor, yanlış sonuçlara yol açabilir.
        data["Tür"] = "Su Faturası"

    # --- Yardımcı Fonksiyonlar ---
    def find_value_near_keywords(keywords, pattern, search_window=120):
        """Anahtar kelimeye yakın ilk eşleşmeyi bulur."""
        for keyword in keywords:
            for match in re.finditer(keyword, lower_text):
                start_pos = match.end()
                search_area = lower_text[start_pos: start_pos + search_window]
                value_match = re.search(pattern, search_area)
                if value_match:
                    return value_match.group(1)
        return None

    def find_largest_amount_near_keywords(keywords, pattern, search_window=180):
        """Anahtar kelimeye yakın en büyük sayıyı bulur."""
        max_amount = -1.0
        found_amount_str = None
        for keyword in keywords:
            for match in re.finditer(keyword, lower_text):
                start_pos = match.end()
                search_area = lower_text[start_pos: start_pos + search_window]
                all_matches = re.findall(pattern, search_area)
                for amount_str in all_matches:
                    try:
                        current_amount = float(amount_str.replace('.', '').replace(',', '.'))
                        if current_amount > max_amount:
                            max_amount = current_amount
                            found_amount_str = amount_str
                    except ValueError:
                        continue
        return found_amount_str

    # --- Son Ödeme Tarihi (yedekli) ---
    date_keywords = ["son ödeme tarihi", "son odeme tarihi", "s.ö.t", "s.o.t", "son ödeme tar"]
    date_pattern = r'(\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4})'

    son_odeme_tarihi = find_value_near_keywords(date_keywords, date_pattern)

    if not son_odeme_tarihi:
        # Metindeki tüm tarih formatına uyan ifadeleri bul
        all_dates = re.findall(date_pattern, lower_text)
        if all_dates:
            latest_date_obj = None
            for date_str in all_dates:
                cleaned_date_str = re.sub(r'\s', '', date_str)
                normalized_date_str = cleaned_date_str.replace('.', '/').replace('-', '/')
                try:
                    current_date_obj = datetime.strptime(normalized_date_str, '%d/%m/%Y')
                    if latest_date_obj is None or current_date_obj > latest_date_obj:
                        latest_date_obj = current_date_obj
                except ValueError:
                    continue
            if latest_date_obj:
                son_odeme_tarihi = latest_date_obj.strftime('%d/%m/%Y')

    if son_odeme_tarihi:
        data["Son Ödeme Tarihi"] = son_odeme_tarihi.replace(" ", "")

    # --- Ödenecek Tutar ---
    amount_keywords = ["ödenecek tutar", "odenecek tutar", "toplam tutar", "kdv dahil toplam", "tutarı", "fatura tutar"]
    tutar = None
    # Hem "1.234,56" hem de "123,45" gibi formatları yakalayan genel regex deseni
    amount_pattern_general = r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})'
    # Adım 1: Anahtar kelimeye yakın ilk tutarı bulmayı dene
    tutar = find_value_near_keywords(amount_keywords, amount_pattern_general)
    # Adım 2: Eğer ilk yöntem başarısız olursa, anahtar kelimeye yakın en BÜYÜK tutarı bulmayı dene
    if not tutar:
        tutar = find_largest_amount_near_keywords(amount_keywords, amount_pattern_general)

    if tutar:
        data["Ödenecek Tutar"] = tutar.replace(" ", "") + " TL"

    # ---  Geçmiş Dönem Borcu ---
    debt_keywords = ["geçmiş dönem borcu", "gecmis donem borcu", "devreden bakiye"]
    debt_pattern_general = r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})'

    gecmis_borc = find_largest_amount_near_keywords(debt_keywords, debt_pattern_general)
    if gecmis_borc:
        data["Geçmiş Dönem Borcu"] = gecmis_borc.replace(" ", "") + " TL"

    return data


if __name__ == "__main__":

    image_file = 'images/elektrik.jpg'
    processed_image = preprocess_image(image_file)

    if processed_image is not None:
        full_text = pytesseract.image_to_string(Image.fromarray(processed_image), lang='tur+eng')

        with open("ocr_output.txt", "w", encoding="utf-8") as f:
            f.write(full_text)

        invoice_details = extract_data_with_logic(full_text)

        print("-------- FATURA BİLGİLERİ -------")
        for key, value in invoice_details.items():
            print(f"{key}: \"{value}\"")
        print("-------------------------------------------------")
    else:
        print(f"HATA: '{image_file}' dosyası bulunamadı veya okunamadı!")
