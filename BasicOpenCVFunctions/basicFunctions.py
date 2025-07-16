import cv2

#imread, imshow ve resize fonksiyonları için birer örnek
img = cv2.imread("ornek.jpg")
resized_img = cv2.resize(img, (500,400))
cv2.imshow("Orijinal Resim",img)
cv2.imshow("Yeniden boyutlandırılmış", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


