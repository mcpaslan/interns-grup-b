1️⃣ Kişi Sayımı (People Counting)

Bu proje, canlı video görüntülerinde kaç kişinin bulunduğunu otomatik olarak sayar.
Kamera görüntüsü üzerinde YOLO gibi nesne tespit modelleri kullanılarak kişiler algılanır. Ardından takip algoritmaları (örneğin ByteTrack veya DeepSORT) sayesinde, her kişiye benzersiz bir ID atanır. Bu ID’ler sayesinde aynı kişi, farklı karelerde yeniden algılansa bile tekrar sayılmaz.
Ekranda hem o anki kişi sayısı hem de zaman içinde pürüzsüzleştirilmiş (EMA yöntemi ile filtrelenmiş) kişi sayısı gösterilir.

2️⃣ Davranış Analizi (Action Recognition)

Sistemin amacı sadece kişi saymak değil, aynı zamanda her kişinin ne yaptığını anlamaktır.
Bunun için iki yöntem kullanılır:

Poz tabanlı analiz (Pose Estimation): İnsan vücudundaki eklem noktaları (örneğin diz, kalça, ayak bileği) tespit edilir. Bu noktalar arasındaki açılar hesaplanarak kişinin “oturduğu”, “yürüdüğü” veya “koştuğu” anlaşılır.

Hız tabanlı analiz: Kişinin merkez noktası kareler arasında ne kadar yer değiştirmişse, bu hızdan yürüyüp yürümediği veya koşup koşmadığı anlaşılır.

Her kişi için, hangi hareketi yaptığı ve o hareketi ne kadar süredir yaptığı ekranda gösterilir.

3️⃣ Çok Kişili Sahnelerde Performans Yönetimi (Performance Optimization in Crowded Scenes)

Bir sahnede az sayıda kişi varken sistem rahat çalışabilir, ancak kalabalık ortamlarda hem kişi sayımı hem de davranış analizi yapmak işlemci (CPU/GPU) açısından daha zor hale gelir.
Bunu çözmek için projede bazı optimizasyonlar uygulanır:

Çözünürlüğü düşürmek (imgsz=416/480) → Daha az piksel işlenir, hız artar.

Kare atlama (vid_stride=2/3) → Her kareyi değil, her 2. veya 3. kareyi işleyerek CPU/GPU yükünü azaltır.

Sınıf filtresi (sadece “person” sınıfı algılanır) → Gereksiz nesneler işlenmez.

Maksimum tespit sayısını sınırlamak (max_det) → Çok kalabalıkta bile sistem belli sayının üzerinde nesne işlemez.

FP16 (yarı hassasiyet) ile GPU hızlandırma → Aynı işi daha az veriyle yaparak hız kazanılır.

Pose hesaplamasını seyrekleştirme (her N. karede poz analizi) → Davranış tespiti için her karede eklem noktaları hesaplamak yerine belli aralıklarla hesaplanır.

Bu sayede sistem, kalabalık sahnelerde bile hem kişi sayımını hem de davranış analizini aynı anda, akıcı şekilde yapabilir.
