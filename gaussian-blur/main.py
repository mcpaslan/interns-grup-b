import cv2 as cv
from matplotlib import pyplot as plt

img=cv.imread('vintage.jpeg')
cv.imshow("middlenightwithcat",img)





# Gaussian blur uygulamaları (sigma değerleri sırasıyla 2, 4, 8)
blur1 = cv.GaussianBlur(img, (0, 0), 2)
blur2 = cv.GaussianBlur(img, (0, 0), 4)
blur3 = cv.GaussianBlur(img, (0, 0), 8)

# RGB formatına çevir
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
blur1 = cv.cvtColor(blur1, cv.COLOR_BGR2RGB)
blur2 = cv.cvtColor(blur2, cv.COLOR_BGR2RGB)
blur3 = cv.cvtColor(blur3, cv.COLOR_BGR2RGB)


#bulanik hale getirilmis resimler bir arada cikti ver (matplotlib ile ciz).
plt.subplot(221),plt.imshow(img), plt.title("original")
plt.subplot(222),plt.imshow(blur1), plt.title("sigma2")
plt.subplot(223),plt.imshow(blur2), plt.title("sigma4")
plt.subplot(224),plt.imshow(blur3), plt.title("sigma8")

# Gaussian blur: X=4, Y=1 → yatay bulanıklık baskın
blur_x1 = cv.GaussianBlur(img, (0, 0), sigmaX=4, sigmaY=1)

# Gaussian blur: X=8, Y=1 → daha fazla yatay bulanıklık
blur_x2 = cv.GaussianBlur(img, (0, 0), sigmaX=8, sigmaY=1)

# Gaussian blur: X=1, Y=4 → dikey bulanıklık baskın
blur_y1 = cv.GaussianBlur(img, (0, 0), sigmaX=1, sigmaY=4)

# Gaussian blur: X=1, Y=8 → daha fazla dikey bulanıklık
blur_y2 = cv.GaussianBlur(img, (0, 0), sigmaX=1, sigmaY=8)



plt.tight_layout()
plt.show()
