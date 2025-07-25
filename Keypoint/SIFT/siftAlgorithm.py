import cv2 as cv

img = cv.imread('ornek.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv.SIFT_create()

# Detect keypoints
kp = sift.detect(gray, None)

# Draw keypoints on the image
img_with_kp = cv.drawKeypoints(gray, kp, img)

# Resized
new_img = cv.resize(img_with_kp, (800,600))

cv.imshow("Deneme", new_img)
cv.waitKey(0)
cv.destroyAllWindows()