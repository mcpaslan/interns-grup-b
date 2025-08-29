import os, re
import numpy as np
import cv2
import streamlit as st
import pytesseract
from pytesseract import Output

# ====== SABİT AYARLAR ======
TESS_EXE  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA  = r"C:\Program Files\Tesseract-OCR\tessdata"
LANG      = "tur+eng"
PSM       = 6
OEM       = 3
DPI       = 300
# ===========================

pytesseract.pytesseract.tesseract_cmd = TESS_EXE
os.environ["TESSDATA_PREFIX"] = TESSDATA

# ---------- Yardımcı Fonksiyonlar ----------
# Streamlit’ten gelen dosya baytlarını BGR görüntüye çevirir (OpenCV formatı).
def bytes_to_bgr(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
# BEYAZ ZEMİN, SİYAH YAZI İÇİN eğer zemin çoğınlukla siyahsa tersine çevirir.
def ensure_black_text_on_white(binary_img):
    white_ratio = cv2.countNonZero(binary_img) / float(binary_img.size)
    return binary_img if white_ratio > 0.5 else (255 - binary_img)
# Basit eğim düzeltme (deskew)
def deskew(binary_img):
    inv = 255 - binary_img
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0: return binary_img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rot = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return rot, angle
# Genel OCR için esnek önişleme (ölçekleme, gürültü azaltma, eşikleme, morfoloji, eğim düzeltme). 
# Görüntüyü merkez etrafında o açı kadar döndürür; döndürülmüş görüntü ve açıyı döndürür.
def preprocess(bgr, upscale=2, denoise_h=12, gaussian=3, use_adaptive=True,
               block_size=31, C=10, morph_open=1, morph_close=1,
               do_dilate=False, do_erode=False, kernel=3, do_deskew=True):
    h, w = bgr.shape[:2]
    if upscale > 1 and max(h, w) < 1600:
        bgr = cv2.resize(bgr, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # eşikleme öncesi gürültü azaltma
    if denoise_h > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
        #gürültü azaltma
    if gaussian and gaussian % 2 == 1:
        gray = cv2.GaussianBlur(gray, (gaussian, gaussian), 0)
# yumuşatma 
    if use_adaptive:
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# morfolojik işlemler için kernel
    K = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    if morph_open > 0:  th = cv2.morphologyEx(th, cv2.MORPH_OPEN, K, iterations=morph_open)
    if morph_close > 0: th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, K, iterations=morph_close)
    if do_dilate:       th = cv2.dilate(th, K, iterations=1)
    if do_erode:        th = cv2.erode(th, K, iterations=1)

    if do_deskew: th, angle = deskew(th)
    else:         angle = 0.0
    final_bin = ensure_black_text_on_white(th)
    return gray, th, final_bin, angle

# --- TCKN yardımcıları (güncel) --
#  11 hane, ilk hane ≠ 0.
def validate_tckn(tckn: str) -> bool:
    if not re.fullmatch(r"[1-9]\d{10}", tckn): return False
    d = [int(x) for x in tckn]
    d10 = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
    d11 = (sum(d[:10])) % 10
    return d[9] == d10 and d[10] == d11
# TCKN için özel önişleme (DÖNDÜRMEDEN, deskew yok).
def preprocess_for_tckn(bgr):
    """Görüntüyü DÖNDÜRMEDEN (deskew yok) TCKN için güçlendirilmiş ikili görsel üretir.
       Döndürme yapılmadığı için bounding box'lar orijinale birebir uyar."""
    h, w = bgr.shape[:2]
    scale = 4 if max(h, w) < 800 else 2
    if scale > 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(bgr)
    gray = r
    gray = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
    Kbig = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, Kbig)
    # Koyu rakamları önplana çıkarır
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return ensure_black_text_on_white(th), scale

SUBS = str.maketrans({"O":"0","o":"0","İ":"1","I":"1","l":"1","ı":"1","S":"5","B":"8","Z":"2","G":"6","T":"7"})
# rakamları dönüştürme tablosu Kimlik noyu okumayı daha sağlamlaştırmak 
def extract_tckn(img_bin):
    cfgs = [
        f"--oem 3 --psm 7 --dpi {DPI} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        f"--oem 3 --psm 6 --dpi {DPI} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ]
    all_candidates = set()
    for cfg in cfgs:
        raw = pytesseract.image_to_string(img_bin, lang='eng', config=cfg).replace("\x0c","")
        norm = raw.translate(SUBS)
        # Boşluk, tire, nokta ile ayrılmış biçimde 11 haneli arama karışan rakamları harflere çevirir 
        for m in re.findall(r"(?<!\d)([1-9][\s\-\.]?\d{2}[\s\-\.]?\d{3}[\s\-\.]?\d{2}[\s\-\.]?\d{2})(?!\d)", norm):
            digits = re.sub(r"\D", "", m)
            if len(digits) == 11:
                all_candidates.add(digits)
                # Sadece rakamları birleştirip 11 haneli olanları topla.
        digits_only = re.sub(r"\D", "", norm)
        for i in range(0, max(0, len(digits_only)-10)):
            s = digits_only[i:i+11]
            if s and s[0] != "0":
                all_candidates.add(s)
    candidates = sorted(all_candidates)
    valid = [x for x in candidates if validate_tckn(x)]
    return candidates, valid
# doğrulanan kimlik numaralarını dönder.
def find_tckn_boxes(img_bin):
    """Tesseract satır bazında TCKN yakalar ve o satırın kutusunu döndürür."""
    df = pytesseract.image_to_data(
        img_bin, lang='eng',
        config=f"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        output_type=Output.DATAFRAME
    )
    df = df[df.conf != -1]
    boxes = []
    # geçersiz satırları atla 
    if df.empty: return boxes
    for _, line in df.groupby(['block_num','par_num','line_num']):
        line = line.dropna(subset=['text'])
        if line.empty: continue
        text = " ".join(str(t) for t in line.text if str(t).strip())
        norm = text.translate(SUBS)
        digits_only = re.sub(r"\D", "", norm)
        # TCKN var mı? kayan pencere ile kontrol et 
        hit = None
        for i in range(0, max(0, len(digits_only)-10)):
            s = digits_only[i:i+11]
            if s and s[0] != "0" and validate_tckn(s):
                hit = s; break
        if hit:
            x1, y1 = int(line['left'].min()), int(line['top'].min())
            x2 = int((line['left']+line['width']).max())
            y2 = int((line['top'] +line['height']).max())
            boxes.append((x1, y1, x2-x1, y2-y1, hit))
    return boxes
# TCKN kutularını orijinal görüntü üzerine çizer ve TC kimlik numarasını yazar.
def draw_tckn_boxes_on_original(orig_bgr, boxes, scale):
    vis = orig_bgr.copy()
    for x, y, w, h, tckn in boxes:
        x0, y0, w0, h0 = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        cv2.rectangle(vis, (x0, y0), (x0+w0, y0+h0), (0,255,0), 2)
        cv2.putText(vis, tckn, (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return vis

# ---------- UI ----------
st.set_page_config(page_title="OCR + Streamlit", page_icon="🧐", layout="wide")


#  1) EN ÜSTE: TC KİMLİK NO OKUMA KUTUSu
# sadece jpg/png yükle 
tckn_box = st.container()
with tckn_box:
    st.subheader("🇹🇷 TR TC Kimlik No Oku")
    colA, colB = st.columns([1,1])
    with colA:
        up_tckn = st.file_uploader("TC Kimlik için görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="tckn_upl")
        st.caption("İstersen TCKN bölgesini elle seç:")
        use_roi_tckn = st.checkbox("TCKN için ROI kullan", value=False, key="tckn_roi_sw")
        rx1 = st.slider("Sol (%)", 0, 95, 10, key="rx1")
        ry1 = st.slider("Üst (%)", 0, 95, 20, key="ry1")
        rx2 = st.slider("Sağ (%)", 5, 100, 60, key="rx2")
        ry2 = st.slider("Alt (%)", 5, 100, 35, key="ry2")
        # BGR formatında yüklenen görsel rgb olarak göster
        if up_tckn:
            bgr_t = bytes_to_bgr(up_tckn.getvalue())
            st.image(cv2.cvtColor(bgr_t, cv2.COLOR_BGR2RGB), caption="Yüklenen Görsel", use_column_width=True)
    with colB:
        if st.button("🔎 TC Kimlik No'yu Tara", use_container_width=True):
            # ROI (Region of Interest) = İlgi Bölgesi
            #Bir görüntünün tamamı yerine, sadece ilgilendiğin dikdörtgen alanı üzerinde işlem yapmana denir.
            if not up_tckn:
                st.warning("Önce bir görsel yükleyin.")
            else:
                src = bgr_t
                if use_roi_tckn:
                    h, w = bgr_t.shape[:2]
                    x1, y1 = int(w*rx1/100), int(h*ry1/100)
                    x2, y2 = int(w*rx2/100), int(h*ry2/100)
                    if x2 > x1 and y2 > y1:
                        src = bgr_t[y1:y2, x1:x2].copy()
                # Ön işleme (döndürme yok) + kutuları bul orijinal üzerine çiz
                bin_t, scale = preprocess_for_tckn(src)
                boxes = find_tckn_boxes(bin_t)
                vis = draw_tckn_boxes_on_original(src, boxes, scale)
                st.image(vis, caption="TCKN Bounding Box (orijinal yön)", use_column_width=True)
                # Metin çıkar ve doğrula
                cands, valid = extract_tckn(bin_t)
                if valid:
                    st.success("Doğrulanan T.C. Kimlik No(ları): **" + "**, **".join(valid) + "**")
                elif cands:
                    st.warning("Aday (doğrulanamadı): " + ", ".join(cands))
                else:
                    st.info("Uygun 11 haneli T.C. Kimlik No bulunamadı.")

st.markdown("---")  # ayırıcı

# 2) GENEL OCR + ROI KISMI 
with st.sidebar:
    st.subheader("Önişleme")
    upscale     = st.slider("Ölçek (küçük görseli büyüt)", 1, 3, 2)
    denoise_h   = st.slider("Gürültü azaltma (h)", 0, 20, 12)
    gaussian    = st.slider("Gaussian kernel", 1, 9, 3, step=2)
    use_adapt   = st.checkbox("Adaptif threshold (kapalıysa Otsu)", value=True)
    block       = st.slider("Adaptif block size", 11, 61, 31, step=2)
    C           = st.slider("Adaptif C", -15, 15, 10)
    kernel      = st.slider("Morfoloji kernel", 1, 9, 3, step=2)
    morph_open  = st.slider("Open (lekeleri sil)", 0, 3, 1)
    morph_close = st.slider("Close (kopuklukları birleştir)", 0, 3, 1)
    do_dilate   = st.checkbox("Dilate (kalınlaştır)", value=False)
    do_erode    = st.checkbox("Erode (incelt)", value=False)
    do_deskew   = st.checkbox("Eğim düzelt (deskew)", value=True)

    st.subheader("ROI (Bölge) Seçimi")
    x1p = st.slider("Sol (%)", 0, 95, 10)
    y1p = st.slider("Üst (%)", 0, 95, 30)
    x2p = st.slider("Sağ (%)", 5, 100, 60)
    y2p = st.slider("Alt (%)", 5, 100, 45)
    only_digits = st.checkbox("ROI: Sadece rakamları oku", value=False)
    only_upper  = st.checkbox("ROI: Büyük harf (TR) odaklı", value=False)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    uploaded = st.file_uploader("Genel OCR için görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="general_upl")
    st.caption("İpucu: önce temiz bir örnekle test edin; sonra gürültülü/eğimli bir örneğe geçin.")
    if uploaded:
        bgr = bytes_to_bgr(uploaded.getvalue())
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Orijinal", use_column_width=True)

        gray, th, final_bin, angle = preprocess(
            bgr, upscale, denoise_h, gaussian, use_adapt, block, C,
            morph_open, morph_close, do_dilate, do_erode, kernel, do_deskew
        )
        st.image(gray, caption="Grileştirilmiş", use_column_width=True, clamp=True)
        st.image(th, caption="Eşiklenmiş", use_column_width=True, clamp=True)
        st.image(final_bin, caption=f"Final (deskew={angle:.2f}°)", use_column_width=True, clamp=True)

        cfg = f"--oem {OEM} --psm {PSM} --dpi {DPI} -c preserve_interword_spaces=1"
        # tesseract için whitelist ekle kelime aralıklarını koru. 
        # whitelist nedir ? Listede olan her şey serbest; olmayan her şey reddedildiği listedir 
        try:
            text = pytesseract.image_to_string(final_bin, lang=LANG, config=cfg)
            text = text.replace("\x0c","").strip()
        except pytesseract.TesseractError as e:
            st.error(f"Tesseract hatası: {e}")
            text = ""
        st.download_button("📥 Metni .txt indir", text.encode("utf-8"),
                           file_name="ocr_cikti.txt", mime="text/plain")

with col_right:
    if 'uploaded' in locals() and uploaded:
        st.subheader("OCR Metni")
        st.text_area("Çıktı", value=text, height=300)
        try:
            data = pytesseract.image_to_data(final_bin, lang=LANG, config=cfg,
                                             output_type=Output.DATAFRAME)
            conf = st.slider("Min. güven (bbox)", 0, 100, 60)
            show_txt = st.checkbox("Kutuların üstüne metni yaz", value=False)
            img_boxed = bgr.copy()
            for i in range(len(data)):
                try: c = float(data["conf"].iloc[i])
                except: c = -1.0
                if c < conf: continue
                x = int(data["left"].iloc[i]); y = int(data["top"].iloc[i])
                w = int(data["width"].iloc[i]); h = int(data["height"].iloc[i])
                cv2.rectangle(img_boxed, (x,y), (x+w, y+h), (0,255,0), 2)
                if show_txt:
                    t = str(data["text"].iloc[i])[:25]
                    cv2.putText(img_boxed, t, (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            st.image(cv2.cvtColor(img_boxed, cv2.COLOR_BGR2RGB), caption="Bounding Boxes", use_column_width=True)
            st.dataframe(data[['text','conf','left','top','width','height']])
        except Exception as e:
            st.caption(f"image_to_data üretilemedi: {e}")
# kutulu görsel ve tabloyu göster.
        st.markdown("### ROI OCR")
        h, w = final_bin.shape[:2]
        x1, y1 = int(w*x1p/100), int(h*y1p/100)
        x2, y2 = int(w*x2p/100), int(h*y2p/100)
        roi_img = final_bin[y1:y2, x1:x2].copy() if (x2>x1 and y2>y1) else final_bin
        st.image(roi_img, caption="Seçilen ROI", use_column_width=True, clamp=True)

        cfg_roi = f"--oem {OEM} --psm 7 --dpi {DPI}"
        if only_digits:
            cfg_roi += " -c tessedit_char_whitelist=0123456789"
        elif only_upper:
            cfg_roi += " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÇĞİÖŞÜ "
        roi_text = pytesseract.image_to_string(roi_img, lang=LANG, config=cfg_roi).replace("\x0c","").strip()
        st.text_area("ROI Metni", value=roi_text, height=150)
