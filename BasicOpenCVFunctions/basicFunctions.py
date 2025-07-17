import cv2

# ------------------------------------------
# OpenCV Temel Fonksiyonlar Örneği
# ------------------------------------------
# Bu örnekte 3 temel fonksiyon kullanılmaktadır:
# 1. imread(): Görüntü dosyasını okuma
# 2. resize(): Görüntünün boyutunu yeniden ayarlama
# 3. imshow(): Görüntüyü ekranda gösterme

# Görüntüyü dosyadan oku
img = cv2.imread("ornek.jpg")

# Görüntüyü 500x400 boyutlarına yeniden boyutlandır
resized_img = cv2.resize(img, (500, 400))

# Orijinal görüntüyü göster
cv2.imshow("Orijinal Resim", img)

# Yeniden boyutlandırılmış görüntüyü göster
cv2.imshow("Yeniden Boyutlandirilmis Resim", resized_img)

# Tuşa basılana kadar bekle
cv2.waitKey(0)

# Tüm pencereleri kapat
cv2.destroyAllWindows()