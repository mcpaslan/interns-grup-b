import cv2

# Görüntüyü oku ve gri tonlamaya çevir
img  = cv2.imread("image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT dedektörünü oluştur ve keypoint+descriptor al
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray, None)

# Keypoint’leri çiz
img_kp = cv2.drawKeypoints(
    img, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imshow("SIFT Keypoints", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
# İkinci görüntüyü oku ve descriptor al
img2  = cv2.imread("image.png")
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher ile eşleştirme yap
bf      = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)

# En iyi 50 eşleşmeyi çiz ve göster
matched = cv2.drawMatches(
    img,  kp1,
    img2, kp2,
    matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imshow("SIFT Matching", matched)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""