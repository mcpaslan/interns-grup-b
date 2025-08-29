import Levenshtein


# --- Fonksiyon Tanımlaması: OCR Düzeltme Motorumuz ---

# find_closest_city adında bir fonksiyon (yani bir görev yapan bir kod bloğu) oluşturuyoruz.
# Bu fonksiyonun görevi, OCR'dan gelen potansiyel olarak hatalı bir kelimeyi alıp,
# bizim "doğru" kabul ettiğimiz kelime listesindeki en benzer kelimeyi bulmaktır.
def find_closest_city(ocr_word, valid_cities, max_distance_threshold=2):
    """
        Args:
        ocr_word (str): OCR'dan gelen ham ve potansiyel olarak hatalı kelime. (Örn: "1STANBL")
        valid_cities (list): Doğru şehir isimlerinin bulunduğu liste. (Örn: ["İSTANBUL", "ANKARA", ...])
        max_distance_threshold (int): Kabul edilebilir maksimum Levenshtein mesafesi.
                                      Bu, bir nevi "güvenilirlik filtresi"dir. Eğer en yakın kelime bile
                                      bu mesafeden daha farklıysa, "bu muhtemelen o kelime değildir" deriz.
    Returns:
        str: En yakın ve en doğru şehir adını döndürür. Eğer güvenilir bir eşleşme bulamazsa "None" (Hiçbir şey) döndürür.
    """

    # --- Hızlı Kontrol: Belki de OCR Doğru Okumuştur? ---

    # İşe başlamadan önce basit bir kontrol yapıyoruz: Acaba OCR'dan gelen kelime,
    # bizim geçerli şehirler listemizde zaten var mı?
    # Eğer varsa, hiç uğraşmaya gerek yok, en doğru sonucu zaten bulduk demektir.
    if ocr_word in valid_cities:
        return ocr_word  # Kelimeyi doğrudan geri döndür ve fonksiyondan çık.

    # --- En İyi Eşleşmeyi Bulmak İçin Hazırlık ---

    # min_distance adında bir değişken oluşturuyoruz. Bu değişken, o anki en iyi
    # (yani en düşük) "farklılık skorunu" aklında tutacak.
    # Başlangıç değeri olarak ona "sonsuzluk" (float('inf')) veriyoruz.
    # Neden? Çünkü karşılaştıracağımız ilk kelimenin farklılık skoru ne olursa olsun,
    # sonsuzluktan daha küçük olacağı için, ilk kelime otomatikman "en iyi adayımız" olacak.
    min_distance = float('inf')

    # best_match adında bir başka değişken oluşturuyoruz. Bu da, en düşük skora sahip
    # olan "en iyi kelime adayını" aklında tutacak. Başlangıçta boş.
    best_match = None

    # --- Karşılaştırma Döngüsü: Tüm Olasılıkları Deneyelim ---

    # Şimdi, 'valid_cities' listemizdeki HER BİR şehir için bir döngü başlatıyoruz.
    # Döngünün her adımında, listedeki bir sonraki şehir 'city' değişkenine atanacak.
    for city in valid_cities:
        # Levenshtein kütüphanesinin 'distance' fonksiyonunu kullanarak,
        # OCR'dan gelen 'ocr_word' ile o anki 'city' arasındaki farklılık skorunu hesaplıyoruz.
        # Örneğin, distance('1STANBL', 'İSTANBUL') sonucu 2 çıkacaktır.
        distance = Levenshtein.distance(ocr_word, city)

        # Şimdi bir karar anı: Hesapladığımız bu yeni 'distance' değeri,
        # şu ana kadar bulduğumuz en düşük değer olan 'min_distance'dan daha mı küçük?
        if distance < min_distance:
            # EĞER EVETSE, daha iyi bir aday bulduk demektir!
            # O zaman en düşük skor rekorumuzu bu yeni değerle güncelliyoruz.
            min_distance = distance
            # Ve en iyi adayımızı da bu yeni şehir olarak güncelliyoruz.
            best_match = city

    # Döngü bittiğinde, 'min_distance' en düşük skoru ve 'best_match' de o skora sahip şehri tutuyor olacak.

    # --- Son Karar: Bulduğumuz Eşleşme Yeterince İyi mi? ---

    # Şimdi "güvenilirlik filtresi"ni kullanma zamanı.
    # Bulduğumuz en iyi eşleşmenin mesafesi ('min_distance'), bizim belirlediğimiz
    # maksimum eşik değerinden ('max_distance_threshold', yani 2'den) küçük veya ona eşit mi?
    if min_distance <= max_distance_threshold:
        # EĞER EVETSE, bu eşleşmeye güvenebiliriz.
        # Ekrana bilgilendirici bir detay mesajı yazdırıyoruz.
        print(f"   (Detay: '{ocr_word}' -> '{best_match}', Mesafe: {min_distance})")
        # Ve bulduğumuz en iyi eşleşmeyi fonksiyonun sonucu olarak döndürüyoruz.
        return best_match
    else:
        # EĞER HAYIRSA, en iyi aday bile bizim için yeterince "benzer" değil demektir.
        # Örneğin, 'DENIZLI' kelimesi 'ADANA' kelimesine çok uzaktır. Bu bir hata olur.
        # Bu yüzden ekrana eşleşme bulamadığımıza dair bir mesaj yazdırıyoruz.
        print(
            f"   (Detay: '{ocr_word}' için güvenilir eşleşme bulunamadı. En yakın aday '{best_match}', Mesafe: {min_distance})")
        # Ve sonuç olarak "None" (Hiçbir şey) döndürüyoruz.
        return None


# --- Senaryoyu Çalıştıralım (Ana Kod Bloğu) ---

# 1. Adım: Bilgi Bankamızı Oluşturalım
# Bu bizim "doğrular listemiz". OCR'dan gelen her kelimeyi bu listedeki
# kelimelerle karşılaştıracağız. Bu liste, bir veritabanından, bir dosyadan veya
# bizim gibi elle yazarak oluşturulabilir.
VALID_TURKISH_CITIES = [
    "İSTANBUL", "ANKARA", "İZMİR", "BURSA", "ANTALYA", "ADANA", "KONYA", "GAZİANTEP", "DENİZLİ"
]

# 2. Adım: Simülasyon Verilerimizi Oluşturalım
# Bu liste, sanki gerçek bir OCR programı farklı faturaları okumuş gibi
# hatalı ve doğru sonuçları içeren bir listedir. Projemizi test etmek için kullanacağız.
ocr_outputs = [
    "1STANBL",  # OCR'ın 'İ' harfini '1' ve 'U' harfini de atladığı bir durum.
    "ANARA",  # 'A' harfinin atlandığı bir durum.
    "İZMIR",  # 'İ' harfinin 'I' olarak yanlış okunduğu çok yaygın bir hata.
    "BÜRSA",  # 'U' harfinin 'Ü' olarak yanlış okunduğu bir hata.
    "ANTALYA",  # Mükemmel bir okuma, hiç hata yok.
    "KONYA",  # Mükemmel bir okuma.
    "GAZIANTEP",  # Yine 'İ' harfinin 'I' olarak okunması hatası.
    "DENIZLI"  # Listemizdeki "DENİZLİ" kelimesine çok yakın, sadece 'İ'/'I' farkı var.
]

# Kullanıcıyı bilgilendirmek için bir başlangıç mesajı yazdırıyoruz.
print("--- OCR Sonrası Şehir Adı Düzeltme İşlemi Başlatıldı ---\n")

# Düzeltilmiş sonuçları saklamak için boş bir liste oluşturuyoruz.
corrected_results = []

# 'ocr_outputs' listesindeki her bir hatalı kelime için döngü başlatıyoruz.
for output in ocr_outputs:
    # Yukarıda tanımladığımız 'find_closest_city' fonksiyonumuzu çağırıyoruz.
    # Fonksiyona o anki hatalı kelimeyi ('output') ve doğrular listemizi ('VALID_TURKISH_CITIES') veriyoruz.
    # Fonksiyonun döndürdüğü sonucu (düzeltilmiş şehir adı veya None) 'corrected_city' değişkenine atıyoruz.
    corrected_city = find_closest_city(output, VALID_TURKISH_CITIES)

    # Sonuçlarımızı daha düzenli tutmak için bir sözlük (dictionary) oluşturup listeye ekliyoruz.
    # Bu, "orijinali neydi, düzeltilmişi ne oldu" bilgisini bir arada tutmamızı sağlar.
    corrected_results.append({
        "original": output,
        "corrected": corrected_city if corrected_city else "Eşleşme Yok"  # Eğer sonuç None ise "Eşleşme Yok" yaz.
    })

# --- Sonuçları Gösterelim ---

# Bütün işlemler bittikten sonra, sonuçları güzel bir formatta ekrana yazdırıyoruz.
print("\n--- Düzeltme Sonuçları ---")
# 'corrected_results' listesindeki her bir sonuç için döngüye giriyoruz.
for result in corrected_results:
    # Her bir orijinal kelimenin nasıl düzeltildiğini kullanıcıya gösteriyoruz.
    print(f"Orijinal OCR Çıktısı: '{result['original']}' -> Düzeltilmiş Sonuç: '{result['corrected']}'")