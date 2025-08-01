import matplotlib.pyplot as plt
import numpy as np


x = np.array([0,6])
y = np.array([0,250])

plt.plot(x,y)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0, 6])
y1 = np.array([0, 250])
plt.plot(x1, y1, label="1. grafik")

x2 = np.array([1, 2, 3, 4])
y2 = np.array([10, 20, 25, 30])
plt.plot(x2, y2, label="2. grafik")

plt.legend()
plt.show()

a = np.array([0,5])
b = np.array([0,40])

plt.plot(a,b,'o')
plt.show()

xpoints = np.array([1,2,3,4])
ypoints = np.array([10,50,30,40])

plt.plot(xpoints,ypoints)
plt.show()

# x olmadan cizim

ypoints = ([3,5,8,15,9])

plt.plot(ypoints)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 1.
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
plt.plot(x, y)
plt.title("Doğrusal Grafik")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")
plt.grid(True)
plt.show()

# 2.
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label="sin(x)")
plt.plot(x, y2, label="cos(x)")
plt.title("Sinüs ve Kosinüs")
plt.legend()
plt.show()

# 3.
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y)
plt.title("Dağılım Grafiği")
plt.show()

# 4.
labels = ['A', 'B', 'C', 'D']
values = [10, 24, 36, 18]
plt.bar(labels, values)
plt.title("Sütun Grafiği")
plt.show()

# 5.
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.title("Histogram")
plt.show()

# 6.
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [35, 25, 25, 15]
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Programlama Dilleri Dağılımı")
plt.show()

# 7.
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(-x)

plt.subplot(2, 1, 1)
plt.plot(x, y1)
plt.title("Sinüs")

plt.subplot(2, 1, 2)
plt.plot(x, y2, color='red')
plt.title("Üstel Azalma")

plt.tight_layout()
plt.show()

# 8.
x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x, y, linestyle='--', color='purple', marker='o')
plt.title("Stil Özelleştirme")
plt.show()

# 9.
x = np.linspace(0, 1, 100)
y1 = np.sin(4 * np.pi * x) * np.exp(-5 * x)
plt.plot(x, y1)
plt.fill_between(x, y1, alpha=0.3)
plt.title("Alan Altı Doldurma")
plt.show()

# 10.
steps = np.random.choice([-1, 1], size=1000)
walk = np.cumsum(steps)
plt.plot(walk)
plt.title("Rastgele Yürüyüş")
plt.show()


turler = ['Aksiyon', 'RPG', 'FPS', 'Spor', 'Strateji', 'Simülasyon']
oyuncu_sayisi = [80, 65, 90, 50, 40, 30]  # Milyon kişi

plt.bar(turler, oyuncu_sayisi, color='skyblue')
plt.title("Oyun Türlerine Göre Oyuncu Sayısı")
plt.xlabel("Oyun Türü")
plt.ylabel("Oyuncu Sayısı (milyon)")
plt.tight_layout()
plt.show()

yillar = [2018, 2019, 2020, 2021, 2022, 2023]
gelir = [138, 150, 174, 180, 192, 205]  # Milyar dolar

plt.plot(yillar, gelir, marker='o')
plt.title("Yıllara Göre Global Oyun Geliri")
plt.xlabel("Yıl")
plt.ylabel("Gelir (milyar $)")
plt.grid(True)
plt.show()

platformlar = ['Mobil', 'PC', 'Konsol']
paylar = [50, 30, 20]

plt.pie(paylar, labels=platformlar, autopct='%1.1f%%', startangle=90)
plt.title("2023 Platformlara Göre Pazar Payı")
plt.show()

motorlar = ['Unity', 'Unreal', 'Godot', 'CryEngine', 'GameMaker']
kullanim_orani = [55, 30, 8, 4, 3]

plt.barh(motorlar, kullanim_orani, color='green')
plt.title("Türkiye'de Oyun Motoru Kullanım Oranları")
plt.xlabel("Kullanım (%)")
plt.tight_layout()
plt.show()

# Her öğrenci bir nokta, x = proje zorluğu, y = kaç kişi seçti
zorluk = np.array([2, 3, 4, 5, 1, 4])
secim_sayisi = np.array([20, 35, 15, 10, 40, 18])
turler = ['Hypercasual', 'Platformer', 'FPS', 'RPG', 'Puzzle', 'Simülasyon']

plt.scatter(zorluk, secim_sayisi)

for i, label in enumerate(turler):
    plt.text(zorluk[i]+0.05, secim_sayisi[i]+0.5, label)

plt.title("Öğrencilerin Tercih Ettiği Oyun Türleri")
plt.xlabel("Zorluk Seviyesi")
plt.ylabel("Kaç Kişi Seçti")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

yas_gruplari = ['7-12', '13-18', '19-25', '26-35']
fps_oran = [10, 40, 55, 30]
puzzle_oran = [50, 30, 15, 10]

x = np.arange(len(yas_gruplari))
width = 0.35

plt.bar(x - width/2, fps_oran, width, label='FPS')
plt.bar(x + width/2, puzzle_oran, width, label='Puzzle')

plt.xticks(x, yas_gruplari)
plt.ylabel("Tercih (%)")
plt.title("Yaşa Göre Oyun Türü Tercihleri")
plt.legend()
plt.tight_layout()
plt.show()

labels = ['Kadın', 'Erkek', 'Diğer']
oranlar = [46, 52, 2]

plt.pie(oranlar, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Cinsiyete Göre Oyuncu Dağılımı (2023)")
plt.axis("equal")
plt.show()

