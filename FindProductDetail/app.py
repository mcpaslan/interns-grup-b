import streamlit as st
import numpy as np
import cv2
from urun_icerik_bul import find_product_details

st.title("Akıllı Ürün Bilgisi Asistanı 📸")
st.write("Ürün fotoğrafını yükleyerek içerik ve besin değerlerini anında öğrenin.")

uploaded_file = st.file_uploader("Bir ürün resmi seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    with st.spinner('Ürün analiz ediliyor...'):
        processed_image, found_products = find_product_details(opencv_image)

    st.subheader("Tespit Sonuçları")

    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption='İşlenmiş Görüntü', use_container_width=True)

    if found_products:
        for product in found_products:
            st.success(f"**Ürün Bulundu:** {product['tam_isim']}")
            st.markdown(f"**Marka:** {product.get('marka', 'N/A')}")

            if product.get('icindekiler'):
                st.markdown("**İçindekiler:**")
                st.text(', '.join(product['icindekiler']))

            if product.get('besin_degerleri'):
                st.markdown("**Besin Değerleri:**")
                st.table(product['besin_degerleri'])
    else:
        st.error("Görüntüde veritabanıyla eşleşen bir ürün bulunamadı.")