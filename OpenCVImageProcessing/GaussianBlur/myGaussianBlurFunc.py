import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


def gaus_kernel(size, sigma=1):
    """
    Belirtilen boyut ve sigma ile 2 boyutlu gaussian kerneli üretilir
    Kernel, toplam 1 olacak şekilde normalize edilir.

    Args:
        size (int): Kernel boyutu (tek sayı olmalı).
        sigma (float): Standart sapması.

    Returns:
        np.ndarray: Normalleştirilmiş Gaussian kerneli.
    """
    kernel = np.zeros((size, size), np.float32)
    center = size // 2
    total = 0.0  # Kernel değerlerinin toplamının tutuldugu degisken

    for x in range(size):
        for y in range(size):
            dx = x - center
            dy = y - center
            kernel[x, y] = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
            total += kernel[x, y]

    kernel /= total  # Normalize islemi
    return kernel


def my_conv_func(image, kernel):
    """
    gönderilen resim ve kernel bilgisi ile konvolüsyon işlemi uygulanır
    Args:
        image : Gri tonlamalı görüntü
        kernel : Konvolüsyon kerneli

    Returns:
        np.ndarray: Konvolüsyon sonucu elde edilen görüntü
    """
    height, width = image.shape
    k = kernel.shape[0]
    pad = k // 2

    # reflect: Aynalama efektidir
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)

    # Her piksel için konvolüsyon uygulanır
    for i in range(height):
        for j in range(width):
            region = padded[i:i + k, j:j + k]  # Çakışan bölge
            output[i, j] = np.sum(region * kernel)

    # Sonuçları 0–255 aralığına kısıtla ve uint8 yap
    return np.uint8(np.clip(output, 0, 255))


# ----------------------- Ana Program ------------ #

image = cv2.imread('ornek.png', cv2.IMREAD_GRAYSCALE)

# Kullanıcıdan kernel boyutunu al
try:
    kernel_size = int(input("Bir tek sayi kernel boyutu girin (5, 7, 9 gibi): "))
    if kernel_size % 2 == 0:
        print("Yalnizca tek sayi girmelisiniz!")
        exit()
except ValueError:
    print("Gecersiz giris. Varsayilan olarak 5 kullaniliyor.")
    kernel_size = 5

# Sigma değerini al
try:
    sigma = float(input("Sigma degerini girin (1.5 gibi): "))
    if sigma <= 0:
        print("Sigma sifirdan buyuk olmali, varsayilan olarak 1 kullaniliyor.")
        sigma = 1.0
except ValueError:
    print("Gecersiz giris. Varsayilan olarak sigma = 1 olarak ayarlandi.")
    sigma = 1.0

# ------------------ Manuel Gaussian Blur ---------------------- #

start_manual = time.time()  # Manuel yazilan gaus fonksiyonu icin zaman hesaplanması yapılır
manual_kernel = gaus_kernel(kernel_size, sigma)  # Kernel oluştur
manual_blur = my_conv_func(image, manual_kernel)  # Konvolüsyon uygula
end_manual = time.time()
manual_duration = end_manual - start_manual  # Süreyi hesapla

# --------------- OpenCV Gaussian Blur --------------------- #

start_cv = time.time()
opencv_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
end_cv = time.time()
opencv_duration = end_cv - start_cv

# ----------------- Sonuçlar ------------ #

print("Elle yazılan Gaussian süresi: ", manual_duration)
print("OpenCV Gaussian blur süresi: ", opencv_duration)

# Görseller burada çizilir ve karşılaştırma yapılır
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Orijinal Resim')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Manuel Gaussian')
plt.imshow(manual_blur, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('OpenCV Gaussian')
plt.imshow(opencv_blur, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
