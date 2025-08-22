import cv2
import pytesseract
import re
import csv
import os
import glob


INPUT_DIR = "fis"
OUTPUT_CSV = "sonuclar.csv"



AMOUNT_RE = re.compile(r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})')
DATE_RE   = re.compile(r'(\d{2}[./]\d{2}[./]\d{4})')

def tr_to_float(x: str) -> float:
    return float(x.replace('.', '').replace(',', '.'))

def normalize(s: str) -> str:
    s = s.upper()
    s = s.replace('0','O').replace('1','I').replace('5','S')
    s = s.replace('İ','I').replace('Ä°','I')
    s = s.replace('TOPLAH','TOPLAM').replace('TOPKOY','TOPKDV').replace('KDY','KDV').replace('KOY','KDV')
    return s

def preprocess(gray):
    g = cv2.bilateralFilter(gray, 9, 75, 75)
    binimg = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 9)
    return binimg

def extract_from_image(path: str):
    img = cv2.imread(path)
    if img is None:
        print("Görsel okunamadı:", path)
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binimg = preprocess(gray)
    cfg = "--oem 1 --psm 6"
    text = pytesseract.image_to_string(binimg, lang="tur+eng", config=cfg)

    # Tarih
    date_match = DATE_RE.search(text)
    date = date_match.group(1) if date_match else None

    # Satır bazlı KDV & TOPLAM
    lines = [l for l in text.splitlines() if l.strip()]
    norm  = [normalize(l) for l in lines]

    kdv, toplam = None, None
    for i,(ln,n) in enumerate(zip(lines,norm)):
        if "KDV" in n:
            cand = AMOUNT_RE.findall(ln)
            if not cand and i+1 < len(lines):
                cand = AMOUNT_RE.findall(lines[i+1])
            if cand: kdv = max(cand, key=tr_to_float).replace('.',',')

        if "TOPLAM" in n or (n.startswith("TOPL") and "KDV" not in n):
            cand = AMOUNT_RE.findall(ln)
            if not cand and i+1 < len(lines):
                cand = AMOUNT_RE.findall(lines[i+1])
            if cand: toplam = max(cand, key=tr_to_float).replace('.',',')

    # Alt ROI yedeği
    if kdv is None or toplam is None:
        h,w = gray.shape
        roi = gray[int(h*0.6):h,0:w]
        roi = cv2.createCLAHE(2.0,(8,8)).apply(roi)
        roi = cv2.threshold(roi,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        txt_roi = pytesseract.image_to_string(roi, lang="tur+eng", config=cfg)
        amts = AMOUNT_RE.findall(txt_roi)
        if amts:
            if kdv is None and "KDV" in normalize(txt_roi):
                kdv = max(amts, key=tr_to_float).replace('.',',')
            if toplam is None:
                toplam = max(amts, key=tr_to_float).replace('.',',')

    return date, kdv, toplam

if __name__ == "__main__":
    # Klasördeki tüm resimleri bul
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp","*.webp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not images:
        print("Hiç görsel bulunamadı! INPUT_DIR yolunu kontrol et.")
        exit()

    rows = []
    for path in images:
        date, kdv, toplam = extract_from_image(path)
        rows.append([os.path.basename(path), date or "", kdv or "", toplam or ""])
        print(f"[{os.path.basename(path)}] Tarih={date}, KDV={kdv}, TOPLAM={toplam}")

    # CSV'ye kaydet
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dosya","tarih","topkdv","toplam"])
        w.writerows(rows)

    print(f"Sonuçlar kaydedildi: {OUTPUT_CSV}")
