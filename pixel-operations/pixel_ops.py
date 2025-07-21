import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Görüntüyü yükle
img = cv.imread("deneme.jpeg")

# Görüntü yüklenemezse uyarı ver ve programı sonlandır
if img is None:
    print("Görüntü yüklenemedi.")
    exit()
else:
    print("Görüntü başarıyla yüklendi:", img.shape)

# Sabit kenar (constant border) için mavi renk tanımı (RGB'ye uygun şekilde yazıldı)
BLUE = [0, 0, 255]  # matplotlib RGB olarak algılayacak

# Farklı kenarlık yöntemleriyle görüntüler oluştur
replicate = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)

# Görüntüleri matplotlib ile çiz (matplotlib RGB bekler ama burada dönüşüm yapılmadı)
plt.subplot(231), plt.imshow(img), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant), plt.title('CONSTANT')

# Tüm grafikleri göster
plt.tight_layout()
plt.show()


