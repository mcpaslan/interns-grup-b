"""
önceki versiyonda resmin kenarlarında meydana gelen piksel kaybını mirroring yöntemiyle ortadan kaldırdım.
optimizasyonu sağlamak için gaussian konvolüsyonu direkt 2 boyutta uygulamak yerine önce yatayda sonra dikeyde 1 boyutlu konvolüsyon yaparak ciddi bir performans artışı elde ettim.
"""
import time
import cv2
import math
import numpy as np

baslangic_zamani = time.time()

def gauss_cekirdek_1d(kernel_size: int, sigma: float):
    """
    kernel_size uzunluğunda 1D Gaussian çekirdeği üretir (normalize edilmiş).
    """
    merkez = kernel_size // 2
    kernel = [0.0] * kernel_size
    toplam_agirlik = 0.0

    # Ham ağırlıkları hesapla
    for i in range(kernel_size):
        x = i - merkez
        w = math.exp(-(x*x) / (2 * sigma * sigma))
        kernel[i] = w
        toplam_agirlik += w

    # Normalize et (toplam = 1)
    for i in range(kernel_size):
        kernel[i] /= toplam_agirlik

    return kernel

def gauss_blur_2d(goruntu: np.ndarray, kernel_size: int, sigma: float):
    """
    Ayrılabilir (separable) 2D Gaussian blur uygular:
      1) Yatayda 1D konvolüsyon
      2) Sonra dikeyde 1D konvolüsyon
    Kenarları mirroring ile halleder.
    """
    yukseklik, genislik, kanal_sayisi = goruntu.shape
    pad = kernel_size // 2
    k1d = gauss_cekirdek_1d(kernel_size, sigma)

    # 1) Görüntüyü yansıtmalı olarak pad'le
    padded = np.pad(
        goruntu,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    # 2) Yatay geçiş
    temp = np.zeros((yukseklik, genislik, kanal_sayisi), dtype=np.float32)
    for y in range(yukseklik):
        for x in range(genislik):
            for kanal in range(kanal_sayisi):
                acc = 0.0
                for i in range(kernel_size):
                    acc += float(padded[y + pad, x + i, kanal]) * k1d[i]
                temp[y, x, kanal] = acc

    # 3) Dikey geçiş için temp'i de pad'le
    padded_temp = np.pad(
        temp,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    # 4) Dikey geçiş ve sonucu uint8'e çevir
    sonuc = np.zeros_like(goruntu, dtype=np.uint8)
    for y in range(yukseklik):
        for x in range(genislik):
            for kanal in range(kanal_sayisi):
                acc = 0.0
                for i in range(kernel_size):
                    acc += padded_temp[y + i, x + pad, kanal] * k1d[i]
                deger = int(acc + 0.5)
                sonuc[y, x, kanal] = 0 if deger < 0 else (255 if deger > 255 else deger)

    return sonuc

# Görüntüyü oku
goruntu = cv2.imread("/Users/yunusemreozturk/Desktop/staj/image.png")

# Ayrılabilir Gaussian blur uygula
bulanik = gauss_blur_2d(
    goruntu,
    kernel_size=25,
    sigma=15.0
)

bitis_zamani = time.time()
print("İşlem süresi:", bitis_zamani - baslangic_zamani)

# Sonucu kaydet ve göster
#cv2.imwrite("/Users/yunusemreozturk/Desktop/img/6.png", bulanik)
cv2.imshow('Orijinal', goruntu)
cv2.imshow('Separable Gaussian Blur', bulanik)
cv2.waitKey(0)
cv2.destroyAllWindows()
