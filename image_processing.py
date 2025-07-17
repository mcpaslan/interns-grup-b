import cv2
import numpy as np

def process_image(path: str):
    # 1) Görüntüyü oku ve BGR’den gri tonlamaya çevir
    img      = cv2.imread(path)                                     # renkli okuma
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # gri tonlamaya dönüştür

    # 2) Global eşikleme (Sabit eşik): piksel ≥ 127 ise beyaz, aksi siyah
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


    # 3) Adaptif eşikleme (Gaussian): her bölgeye göre farklı eşik hesaplar (bölgedeki threshold o bölgenin gaussain ağırlıklı ortalaması eksi C)
    adapt     = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # çevredeki Gaussian ağırlıklı ortalama
        cv2.THRESH_BINARY,               # ikili sonuç
        blockSize=11,                    # bölge boyutu
        C=2                              # eşikten çıkarılacak sabit
    )

    # 4) Adaptif eşikleme (Mean): her bölgeye göre farklı eşik hesaplar (bölgedeki threshold o bölgenin ortalaması eksi C)
    adapt2 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,  # çevredeki ortalama
        cv2.THRESH_BINARY,  # ikili sonuç
        blockSize=11,  # bölge boyutu
        C=2  # eşikten çıkarılacak sabit
    )

    # 5) Canny kenar algılama: önce gürültü azaltmak için blur, sonra kenar tespiti
    blur      = cv2.GaussianBlur(gray, (5,5), sigmaX=1.4)           # hafif bulanıklaştırma
    canny     = cv2.Canny(blur, 50, 150)                             # düşük-yüksek eşikli kenar

    # 6) Morfolojik işlemler için yapı elemanı (kernel)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # eliptik şekilli matris

    # 7) Erosion: her beyaz pikselde kernel matrisinin üzerinde döner,
    #    eğer kernel içindeki pikselin tüm çaprazındaki çevresel pikseller beyazsa,
    #    o merkez piksel beyaz kalır; aksi halde siyaha dönüşür (yakınlaştırma min filtresi)
    eroded    = cv2.erode(binary, kernel, iterations=1)

    # 8) Dilation: her siyah pikselde kernel matrisinin üzerinde döner,
    #    kernel içindeki bir beyaz nokta bulursa o merkez pikseli beyaz yapar
    #    (genişletme max filtresi, boşlukları doldurur)
    dilated   = cv2.dilate(binary, kernel, iterations=1)

    # 9) Opening: önce erode sonra dilate uygular,
    #    böylece önce küçük beyaz lekeleri siler, sonra kalan alanı eski haline döndürür
    opened    = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # 10) Closing: önce dilate sonra erode uygular,
    #     böylece önce boşlukları doldurur, sonra kenarları eski haline getirir
    closed    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 11) Sonuçları göster
    cv2.imshow("Orijinal", img)                                      # orijinal renkli
    cv2.imshow("Gri Ton", gray)                                      # gri düzlem
    cv2.imshow("Binary (Sabit Esik)", binary)                        # sabit eşik
    cv2.imshow("Adaptif Esikleme", adapt)                            # yerel eşik
    cv2.imshow("Adaptif Esikleme (Mean)", adapt2)                            # yerel eşik
    cv2.imshow("Canny Kenar", canny)                                 # kenar haritası
    cv2.imshow("Erosion", eroded)                                    # gürültü silinmiş
    cv2.imshow("Dilation", dilated)                                # boşluklar kapatılmış
    cv2.imshow("Opening", opened)                                    # küçük beyaz lekeler temiz
    cv2.imshow("Closing", closed)                                 # küçük siyah delikler dolmuş

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/Users/yunusemreozturk/Desktop/deneme.png"
    process_image("/Users/yunusemreozturk/Desktop/staj/balon.png")
