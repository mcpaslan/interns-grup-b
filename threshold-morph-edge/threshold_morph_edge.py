import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def apply_thresholds(gray_img):
    _, binary_thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    otsu_thresh_val, otsu_thresh = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2)
    return binary_thresh, otsu_thresh, adaptive_thresh, otsu_thresh_val

def auto_canny(img, sigma=0.33):
    med = np.median(img)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edged = cv2.Canny(img, lower, upper)
    return edged

def morphological_operations(binary_img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary_img, kernel, iterations=1)
    dilation = cv2.dilate(binary_img, kernel, iterations=1)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return erosion, dilation, opening, closing

def plot_images(title_list, image_list, rows=2, cols=4):
    plt.figure(figsize=(20, 10))
    for i, (title, img) in enumerate(zip(title_list, image_list)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----------------- Ana Akış -----------------
image_name = "input1.jpeg"  # Güncellenmiş dosya adı
original, gray = load_image(image_name)

# Threshold yöntemleri
binary, otsu, adaptive, otsu_val = apply_thresholds(gray)

# Otomatik Canny kenar algılama
edges = auto_canny(gray)

# Morfolojik işlemler (Otsu ile binary'den)
erosion, dilation, opening, closing = morphological_operations(otsu)

# Görselleştirme
titles = [
    'Grey', 'Thresholding (127)',
    f'Threshold (val={otsu_val:.2f})', 'Adaptive Threshold',
    'Canny', 'Erosion',
    'Dilation ', 'Closing'
]

images = [gray, binary, otsu, adaptive, edges, erosion, dilation, closing]

plot_images(titles, images)
