import os, re, json, unicodedata
import numpy as np
import cv2
import streamlit as st
import pytesseract
from pytesseract import Output
import pandas as pd

# ---- Opsiyonel kütüphaneler ----
try:
    from rapidfuzz import process, fuzz
except Exception:
    process = fuzz = None

try:
    import spacy
    _nlp = spacy.load("xx_ent_wiki_sm")
except Exception:
    _nlp = None

try:
    import camelot
except Exception:
    camelot = None

# ---- Tesseract Yolları (gerekirse değiştir) ----
TESS_EXE  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA  = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = TESS_EXE
os.environ["TESSDATA_PREFIX"] = TESSDATA

# ===== Yardımcılar =====
def bytes_to_bgr(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def ocr_image(img, lang="tur+eng", psm=6, oem=3, dpi=300, config_extra=""):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cfg = f"--oem {oem} --psm {psm} --dpi {dpi} {config_extra}"
    txt = pytesseract.image_to_string(gray, lang=lang, config=cfg)
    return txt.replace("\x0c","").strip(), gray

TR_MAP = {"’":"'", "“":'"', "”":'"', "–":"-", "—":"-",
          "ﬁ":"fi", "ﬂ":"fl", "·":".", "•":"-", "●":"-",
          "¼":"1/4", "½":"1/2", "¾":"3/4"}

def ocr_clean(text:str, keep_punct=True)->str:
    t = unicodedata.normalize("NFKC", text or "")
    for k,v in TR_MAP.items(): t = t.replace(k,v)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if not keep_punct:
        t = re.sub(r"[^\w\sçğıöşüİÇĞÖŞÜ.,:/\-@+]", " ", t)
    # "A d r e s" → "Adres"
    t = re.sub(r"(?<=\w) (?=\w(?: \w){2,})", "", t)
    return t.strip()

TR_DOMAIN_VOCAB = [
    "TÜRKİYE","CUMHURİYETİ","ADI","SOYADI","TC","T.C.","KİMLİK","NUMARASI",
    "DOĞUM","TARİHİ","ADRES","KURUM","FATURA","TUTAR","FORM","BAŞVURU",
    "HASTA","DOKTOR","REÇETE","SERİ","NO","E-POSTA","TELEFON","VERGİ","DAİRESİ"
]

def correct_with_vocab(text:str, vocab:list, score_cut=90):
    if process is None or fuzz is None:
        return text  # RapidFuzz yoksa dokunma
    tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", text)
    rep = {}
    for tok in set(tokens):
        if tok.isupper() or len(tok) < 3:
            match, score, _ = process.extractOne(tok, vocab, scorer=fuzz.WRatio)
            if score >= score_cut:
                rep[tok] = match
    def _repl(m):
        w = m.group(0)
        return rep.get(w, w)
    return re.sub(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", _repl, text)

RE_TCKN   = re.compile(r"(?<!\d)([1-9]\d{10})(?!\d)")
RE_DATE   = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{4})")
RE_PHONE  = re.compile(r"0?\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}")
RE_MAIL   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_NAME   = re.compile(r"([A-ZÇĞİÖŞÜ]{2,}(?:\s+[A-ZÇĞİÖŞÜ]{2,}){1,3})")

def extract_fields_from_text(text:str):
    out = {
        "tckn": list({m.group(1) for m in RE_TCKN.finditer(text)}),
        "tarih": list({m.group(1) for m in RE_DATE.finditer(text)}),
        "telefon": list({re.sub(r"\s+","",m.group(0)) for m in RE_PHONE.finditer(text)}),
        "email": list({m.group(0) for m in RE_MAIL.finditer(text)}),
    }
    cand_names = [m.group(1) for m in RE_NAME.finditer(text)]
    out["ad_soyad"] = [n for n in cand_names if len(n.split())>=2 and len(n)<=60]
    return out

def nlp_entities(text:str):
    ents = []
    if _nlp is None: return ents
    doc = _nlp(text)
    for e in doc.ents:
        ents.append({"text": e.text, "label": e.label_})
    return ents

def extract_table_from_image(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,5)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, hkernel, iterations=1)
    vertical   = cv2.morphologyEx(th, cv2.MORPH_OPEN, vkernel, iterations=1)
    grid = cv2.add(horizontal, vertical)
    cnts,_ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h>500 and w>20 and h>15:
            cells.append((y,x,w,h))
    cells = sorted(cells)[:200]
    rows, row, last_y = [], [], None
    for (y,x,w,h) in cells:
        if last_y is None or abs(y-last_y)<10:
            row.append((x,y,w,h)); last_y=y
        else:
            rows.append(sorted(row)); row=[(x,y,w,h)]; last_y=y
    if row: rows.append(sorted(row))
    data=[]
    for r in rows:
        row_txt=[]
        for (x,y,w,h) in r:
            roi = gray[y:y+h, x:x+w]
            binroi = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            t = pytesseract.image_to_string(binroi, lang="tur+eng", config="--oem 3 --psm 6 --dpi 300").replace("\x0c","").strip()
            row_txt.append(ocr_clean(t))
        if any(s for s in row_txt): data.append(row_txt)
    if not data: return None
    m = max(len(r) for r in data)
    for r in data: r += [""]*(m-len(r))
    return pd.DataFrame(data)

# ===== UI =====
st.set_page_config(page_title="OCR Suite (28–34)", page_icon="🧐", layout="wide")


tabs = st.tabs([
    "28) TR OCR",
    "29) Temizleme",
    "30) Post-Processing",
    "31) Alan Çıkarımı",
    "32) OCR + NLP",
    "33) Form Otomasyonu",
    "34) Tablo Tanıma",
])

# ---- 28 ----
with tabs[0]:
    st.subheader("28) OCR: Türkçe Metin Tanıma")
    c1, c2 = st.columns(2)
    with c1:
        img_file = st.file_uploader("Görsel yükle (PNG/JPG)", type=["png","jpg","jpeg"], key="i28")
        lang = st.text_input("Dil (lang)", value="tur+eng", key="lang28")
        psm = st.selectbox("PSM", [3,4,6,7,11,12,13], index=2, key="psm28")
    with c2:
        oem = st.selectbox("OEM", [0,1,2,3], index=3, key="oem28")
        dpi = st.slider("DPI", 100, 600, 300, 50, key="dpi28")
    if img_file:
        bgr = bytes_to_bgr(img_file.getvalue())
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Görsel", use_column_width=True)
        text, gray = ocr_image(bgr, lang=lang, psm=psm, oem=oem, dpi=dpi)
        st.image(gray, caption="Grayscale", use_column_width=True, clamp=True)
        st.text_area("OCR Metni", value=text, height=280, key="txt28")
        st.download_button("📥 .txt indir", data=text.encode("utf-8"), file_name="ocr_tr.txt", mime="text/plain")
    else:
        st.info("Bir görsel yükleyin.")

# ---- 29 ----
with tabs[1]:
    st.subheader("29) OCR Sonuçlarını Temizleme")
    img_file = st.file_uploader("Görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="i29")
    keep_punct = st.checkbox("Noktalama işaretlerini koru", value=True, key="kp29")
    if img_file:
        bgr = bytes_to_bgr(img_file.getvalue())
        text,_ = ocr_image(bgr)
        st.text_area("Ham OCR", value=text, height=180, key="h29")
        cleaned = ocr_clean(text, keep_punct=keep_punct)
        st.text_area("Temizlenmiş", value=cleaned, height=220, key="c29")
        st.download_button("📥 Temiz .txt", data=cleaned.encode("utf-8"), file_name="ocr_clean.txt")
    else:
        st.info("Görsel yükleyin.")

# ---- 30 ----
with tabs[2]:
    st.subheader("30) OCR Post-Processing & Düzeltme (Levenshtein)")
    c1, c2 = st.columns(2)
    with c1:
        img_file = st.file_uploader("Görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="i30")
        score_cut = st.slider("Benzerlik eşiği", 70, 100, 90, 1, key="sc30")
        user_vocab = st.text_area("Ek sözlük (satır başına kelime)", value="", key="voc30")
        vocab = TR_DOMAIN_VOCAB + [w.strip() for w in user_vocab.splitlines() if w.strip()]
    with c2:
        if img_file:
            bgr = bytes_to_bgr(img_file.getvalue())
            text,_ = ocr_image(bgr)
            st.text_area("Ham OCR", value=text, height=250, key="h30")
            fixed = correct_with_vocab(text, vocab, score_cut=score_cut)
            if process is None:
                st.warning("RapidFuzz kurulu değil: `pip install rapidfuzz`")
            st.text_area("Düzeltilmiş", value=fixed, height=250, key="f30")
            st.download_button("📥 Düzeltilmiş .txt", data=fixed.encode("utf-8"), file_name="ocr_fixed.txt")
        else:
            st.info("Görsel yükleyin.")

# ---- 31 ----
with tabs[3]:
    st.subheader("31) OCR ile Alan Bazlı Veri Çıkarımı")
    t1, t2 = st.tabs(["Görüntüden OCR", "Metinden Doğrudan"])
    with t1:
        img_file = st.file_uploader("Görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="i31")
        if img_file:
            bgr = bytes_to_bgr(img_file.getvalue())
            text,_ = ocr_image(bgr)
            st.text_area("OCR Metni", value=text, height=180, key="m31")
            st.json(extract_fields_from_text(text))
        else:
            st.info("Görsel yükleyin.")
    with t2:
        txt = st.text_area("Metin yapıştır", height=180, key="pt31")
        if st.button("Alanları çıkar", key="btn31"):
            st.json(extract_fields_from_text(txt))

# ---- 32 ----
with tabs[4]:
    st.subheader("32) OCR + NLP ile Anlam Çıkarımı")
    img_file = st.file_uploader("Görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="i32")
    if img_file:
        bgr = bytes_to_bgr(img_file.getvalue())
        text,_ = ocr_image(bgr)
        st.text_area("OCR Metni", value=text, height=220, key="m32")
        ents = nlp_entities(text)
        if _nlp is None:
            st.warning("SpaCy modeli yok: `pip install spacy && python -m spacy download xx_ent_wiki_sm`")
        else:
            st.dataframe(pd.DataFrame(ents))
    else:
        st.info("Görsel yükleyin.")

# ---- 33 ----
with tabs[5]:
    st.subheader("33) OCR ile Form Otomasyonu")
    img_file = st.file_uploader("Doldurulmuş form görseli (PNG/JPG)", type=["png","jpg","jpeg"], key="i33")
    if img_file:
        bgr = bytes_to_bgr(img_file.getvalue())
        text,_ = ocr_image(bgr)
        st.text_area("OCR Metni", value=text, height=220, key="m33")
        fields = {
            "tckn": (RE_TCKN.search(text) or [None])[0] if RE_TCKN.search(text) else None,
            "tarih": (RE_DATE.search(text) or [None])[0] if RE_DATE.search(text) else None,
            "telefon": re.sub(r"\s+","", (RE_PHONE.search(text) or [None])[0]) if RE_PHONE.search(text) else None,
            "email": (RE_MAIL.search(text) or [None])[0] if RE_MAIL.search(text) else None,
        }
        st.json(fields)
        st.download_button("📥 Form JSON", data=json.dumps(fields, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="form_fields.json", mime="application/json")
        df = pd.DataFrame([fields])
        st.download_button("📥 Form CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="form_fields.csv", mime="text/csv")
    else:
        st.info("Görsel yükleyin.")

# ---- 34 ----
with tabs[6]:
    st.subheader("34) OCR + Tablo Tanıma")
    st.markdown("**PDF (Camelot)**")
    pdf = st.file_uploader("PDF yükle", type=["pdf"], key="pdf34")
    if pdf:
        if camelot is None:
            st.error("camelot-py yok: `pip install camelot-py[cv]` (Ghostscript gerekebilir).")
        else:
            tmp = "tmp_tab.pdf"
            with open(tmp, "wb") as f: f.write(pdf.read())
            try:
                tables = camelot.read_pdf(tmp, pages="1-end", flavor="lattice")
                st.write(f"Bulunan tablo sayısı: {tables.n}")
                for i,t in enumerate(tables):
                    df = t.df
                    st.dataframe(df)
                    st.download_button(f"CSV indir (Tablo {i+1})", df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"tablo_{i+1}.csv")
            except Exception as e:
                st.error(f"Camelot hata: {e}")
    st.markdown("---")
    st.markdown("**Görsel (deneysel grid)**")
    img = st.file_uploader("Görsel (PNG/JPG)", type=["png","jpg","jpeg"], key="img34")
    if img:
        bgr = bytes_to_bgr(img.getvalue())
        df = extract_table_from_image(bgr)
        if df is not None:
            st.dataframe(df)
            st.download_button("CSV indir", df.to_csv(index=False).encode("utf-8"),
                               file_name="tablo_img.csv")
        else:
            st.info("Çizgili tablo bulunamadı ya da OCR yetersiz.")
