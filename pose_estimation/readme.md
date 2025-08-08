# Squat Tracker

MediaPipe kullanarak **videoda squat (çömelme) tespiti ve sayımı** yapan, her pozda (Standing/Squatting) **ne kadar süre kalındığını** ölçüp **CSV/JSON** olarak dışa aktaran bir Python uygulaması.

---

##  Özellikler

- Gerçek zamanlı **squat tespiti ve sayımı**  
- **State-machine** mantığıyla sağlam tekrar sayımı (down→up tamamlanınca artar)  
- Anlık **poz etiketi**: `Standing` ↔ `Squatting`  
- **Davranış logu**: her poz için **başlangıç/bitiş zamanı** ve **süre (sn)**  
- Otomatik **CSV** ve **JSON** çıktı

---

##  Gereksinimler

- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [NumPy](https://pypi.org/project/numpy/)

> Öneri: Sanal ortam (virtualenv/venv) kullanın.

---

##  Kurulum

1) Depoyu/Projenizi klasöre alın (veya kendi dosyalarınızı oluşturun).  
2) Sanal ortam oluşturun ve etkinleştirin:
```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\activate       # Windows
```
3) Bağımlılıkları kurun:
```bash
pip install opencv-python mediapipe numpy
```



---

##  Proje Yapısı (Önerilen)

```
.
├── main.py               # Girişi çalıştıran dosya
├── squat_detector.py     # SquatDetector sınıfı (algoritma + logging)
└── README.md
```


---

##  Kullanım

```bash
python main.py /tam/yol/squat.mp4
```

- Pencere açıkken **q** tuşuyla çıkış yapabilirsiniz.
- Video bittiğinde şu dosyalar proje dizinine kaydedilir:
  - `squat_behavior.csv`
  - `squat_behavior.json`



---

##  Nasıl Çalışır?

1. **MediaPipe Pose** her karede 33 eklem noktası çıkarır.  
2. **Diz açısı** (kalça–diz–ayak bileği) her iki bacak için hesaplanır ve **ortalaması** alınır.  
3. **State-machine**:
   - `Standing → Squatting`: ortalama diz açısı `DOWN_THRESH` altına indiğinde
   - `Squatting → Standing`: ortalama diz açısı `UP_THRESH` üstüne çıktığında → **1 tekrar** sayılır
4. Poz değişimlerinde **zaman damgası** alınır; **başlangıç/bitiş** ve **süre (sn)** loga eklenir.  
5. İşlem bitince loglar **CSV/JSON** olarak yazılır.

---

##  Parametreler

`SquatDetector(down_thresh=70, up_thresh=160, min_det_conf=0.5, min_track_conf=0.5)`

- `down_thresh` : Squat **başlangıç** eşiği (°) — daha küçük değer **daha derin** çömelme ister.  
- `up_thresh`   : Squat **bitiş** eşiği (°) — daha büyük değer **daha dik** kalkışı bekler.  
- `min_det_conf`/`min_track_conf`: MediaPipe tespit/izleme güven eşikleri.

> Farklı kişi, açı ve kamera konumlarında eşikleri **kalibre** etmeniz gerekebilir.

---

##  Çıktı Örnekleri

**CSV (`squat_behavior.csv`)**

| pose      | start                      | end                        | duration_s |
|-----------|----------------------------|----------------------------|------------|
| Standing  | 2025-08-07T14:00:01.123    | 2025-08-07T14:00:03.456    | 2.333      |
| Squatting | 2025-08-07T14:00:03.456    | 2025-08-07T14:00:05.789    | 2.333      |

**JSON (`squat_behavior.json`)**
```json
[
  {
    "pose": "Standing",
    "start": "2025-08-07T14:00:01.123",
    "end": "2025-08-07T14:00:03.456",
    "duration_s": 2.333
  },
  {
    "pose": "Squatting",
    "start": "2025-08-07T14:00:03.456",
    "end": "2025-08-07T14:00:05.789",
    "duration_s": 2.333
  }
]
```

---



