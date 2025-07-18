import streamlit as st

import streamlit as st

st.title("📸 Görüntü İşleme Sunumu")
st.header("1. Python Temelleri")
st.write("Python'da değişken, fonksiyon, döngü gibi yapılar...")

st.code("""
def topla(a, b):
    return a + b
""", language='python')

st.image("ornek_resim.png", caption="Orijinal Görüntü")
