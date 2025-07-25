import cv2
"""
# Görüntüleri oku ve griye çevir
img1 = cv2.imread("image.png")
img2 = cv2.imread("image.png")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB dedektörünü oluştur
#    nfeatures: kaç adet en güçlü keypoint tutulacak
orb = cv2.ORB_create(nfeatures=500)

# Keypoint ve descriptor hesapla
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Tek görselde keypoint’leri çiz ve göster
img1_kp = cv2.drawKeypoints(
    img1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imshow("ORB Keypoints (Image 1)", img1_kp)



# Descriptor eşleştirme (Hamming + crossCheck)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)

# İlk 50 eşleşmeyi çiz
matched_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imshow("ORB Matches", matched_img)

# Pencereleri beklet ve kapat
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def detect_orb_keypoints(image_path: str, nfeatures: int = 500):
    """
    Verilen resimde ORB algoritmasıyla keypoint tespiti yapar
    ve bulunan keypoint’leri çizerek ekranda gösterir.
    """
    # Görüntüyü oku ve gri tonlamaya çevir
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise IOError(f"Resim bulunamadı: {image_path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ORB nesnesini oluştur
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Keypoint ve descriptor hesapla (descriptors burada kullanılmayacak)
    keypoints, _ = orb.detectAndCompute(img_gray, None)

    # Keypoint’leri çiz
    img_with_kp = cv2.drawKeypoints(
        img_color,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Sonucu göster
    cv2.imshow("ORB Keypoints", img_with_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_orb_keypoints("image.png", nfeatures=300)

