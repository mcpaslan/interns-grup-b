1) Genel OCR ve TCKN

Görsellerden Türkçe+İngilizce metin tanıma (PSM/OEM/DPI ayarlı).

Önişleme zinciri: gri, gürültü azaltma, adaptive/Otsu threshold, morfoloji, isteğe bağlı deskew.

ROI seçimiyle belli bir alanı hızlı ve daha doğru okuma.

TCKN akışı: özel önişleme (deskew yok), olası karışan karakterleri eşleme, regex ile aday toplama, d10/d11 kontrol toplamı ile doğrulama.

Kelime kutuları (bounding box) ve güven skorlarıyla görsel üstünde işaretleme.

2) Metin Temizleme ve Düzeltme

Unicode normalizasyonu, sık görülen tipografik karakterlerin sadeleştirilmesi.

Boşluk/çizgi/artık karakterlerin temizlenmesi.

(Opsiyonel) RapidFuzz ile sözlüklere göre yazım yakınsama düzeltmesi.

3) Alan (Field) Çıkarımı

Metinden TCKN, tarih, telefon, e-posta gibi yapısal alanların regex ile yakalanması.

Büyük harf bloklarından ad-soyad adaylarının çıkarılması.

4) OCR + NLP (Varlık Tanıma)

SpaCy ile metin üzerinde kişi/kurum/yer vb. varlık etiketleri (model mevcutsa).

Metin sonucu ve varlık listelerini tablo halinde görüntüleme.

5) Form Otomasyonu

OCR metninden alanları toplayıp JSON/CSV çıktıları üretme.

Basit form doldurma/entegrasyon senaryolarına temel oluşturma.

6) Tablo Tanıma (PDF/Görsel)

Camelot ile PDF’lerden tablo çıkarımı ve CSV indirme.

Görsellerde çizgisel grid algılama (deneysel) + hücre bazlı OCR ile tablo oluşturma.

Arayüz & Kullanım

Streamlit sekmeleri (28–34):

28) TR OCR

Temizleme

Post-Processing

Alan Çıkarımı

OCR + NLP

Form Otomasyonu

Tablo Tanıma

Her sekmede örnek dosya yükleme, ayar slider’ları ve indirme butonları bulunur.
