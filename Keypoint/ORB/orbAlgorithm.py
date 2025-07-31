import cv2

# Görseli gri tonlamalı olarak oku
image = cv2.imread("ornek.jpeg", cv2.IMREAD_GRAYSCALE)

# ORB dedektörü oluştur
orb = cv2.ORB_create(nfeatures=500)  # Maksimum 500 anahtar nokta

# Anahtar noktaları ve descriptor'ları tespit et
keypoints, descriptors = orb.detectAndCompute(image, None)

# Anahtar noktaları görsele çiz
image_with_keypoints = cv2.drawKeypoints(
    image, keypoints, None, color=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# Sonucu göster
cv2.imshow("ORB Anahtar Noktalari", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
