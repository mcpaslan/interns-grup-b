# interns-grup-b
CNN(Convolutional Neural Network)
CNN NEDİR?:
CNN, özellikle görüntü işleme ve bilgisayarla görme (computer vision) problemlerinde çok başarılı olan bir yapay sinir ağı mimarisidir
 
En basit şekilde bir CNN modeli görseldeki gibi özetlenebilir. Bir giriş görüntüsü (Input), evrişim katmanı (CONV), ortaklama (pooling) ve tam bağlantı (fully connected — FC) katmanlarından (layers) oluşur. Şimdide bu katmanları inceleyelim.
1-)Convolution (evrişim katmanı)
Görüntüden özellik çıkarır. Kernel (filtre) ile görüntü üzerinde kayan pencere yöntemi uygulanır.
 
S(i,j)=m∑n∑I(i+m,j+n)⋅K(m,n)
 CNN’de tamamen rastgele filtreler üretiyoruz, bu rastgele ürettiğimiz filtreleri resmimize uygulayarak resmin belirli bölgelerinden özellik çıkarımı yapmaya çalışıyoruz.


1 2 3 0 1  
0 1 2 3 1         
3 1 0 2 2       5*5 lik görüntümüz 
1 2 1 0 0  
0 1 3 1 2

0 1 0  
1 -4 1    3*3 lük filtremiz
0 1 0

1 2 3  
0 1 2     sadece 1 bölgeye uygulanan filtrelememiz 
3 1 0
S(i,j)=(0⋅1)+(1⋅2)+(0⋅3)+(1⋅0)+(−4⋅1)+(1⋅2)+(0⋅3)+(1⋅1)+(0⋅0)=2−4+2+1=1 bu formül ile bulunuyor

 


2-)Ortaklama (POOLİNG): Pooling, belirli bir boyutta pencere ile görüntü veya feature map üzerinde gezinerek, o pencere içindeki değeri bir özet değere indirger.
Pooling türleri
Max Pooling	Pencere içindeki en büyük değeri alır. En yaygın olanıdır.
Average Pooling	Pencere içindeki ortalama değeri alır.
Min Pooling (nadir)	En küçük değeri alır.
Y(i,j)=0≤m<fmax0≤n<fmaxX(i⋅s+m,j⋅s+n)
  Girdi matris 
  MAX POOLİNG 
 
•	3-) Tam Bağlantı (Fully-Connected — FC):
Her nöron, önceki katmandaki tüm nöronlara bağlıdır.
CNN’de:
•	Conv ve pooling katmanları özellikleri çıkarır.
•	FC katmanı bu özellikleri alır ve sınıflandırma, regresyon, karar verme gibi son çıktıyı üretir.
z=w1x1+w2x2+⋯+wnxn+b=wT⋅x+b

Diyelim Conv + Pooling sonucu şu vektör çıktı:
x=[0.2, 0.5, 0.1]\mathbf{x} = [0.2,\ 0.5,\ 0.1]x=[0.2, 0.5, 0.1] 
Ve FC katman ağırlıkları:
w=[0.3, −0.7, 0.5],b=0.2\mathbf{w} = [0.3,\ -0.7,\ 0.5],\quad b = 0.2w=[0.3, −0.7, 0.5],b=0.2 
Sonuç:
z=0.3⋅0.2+(−0.7)⋅0.5+0.5⋅0.1+0.2=0.06−0.35+0.05+0.2=−0.04z = 0.3\cdot0.2 + (-0.7)\cdot0.5 + 0.5\cdot0.1 + 0.2 = 0.06 - 0.35 + 0.05 + 0.2 = -0.04z=0.3⋅0.2+(−0.7)⋅0.5+0.5⋅0.1+0.2=0.06−0.35+0.05+0.2=−0.04 
Aktivasyon fonksiyonu (örnek: ReLU):
a=max⁡(0,−0.04)=0a = \max(0, -0.04) = 0a=max(0,−0.04)=0
 
4-) Activation:
Aktivasyon fonksiyonu, nöronun ağırlıklı girişlerinin toplamını alır ve çıktının aktif mi pasif mi olacağına karar verir.
z=wT⋅x+b
Birkaç tane yaygın activation fonksiyonu var 
🔹 ReLU (Rectified Linear Unit)
Negatif girişleri sıfıra indirgerken, pozitif değerleri olduğu gibi geçirir.
Bu sayede hesaplamalar daha hızlı olur ve öğrenme süreci verimli ilerler.
Formül:
f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x) 
________________________________________
🔹 Sigmoid
Girdiyi 0 ile 1 arasındaki bir değere dönüştürür.
Özellikle iki sınıflı (binary) sınıflandırma problemlerinde kullanılır.
Formül:
f(x)=11+e−xf(x) = \frac{1}{1 + e^{-x}}f(x)=1+e−x1 
________________________________________
🔹 Tanh (Hiperbolik Tanjant)
Girdiyi -1 ile 1 arasına çeker. Sigmoid’e göre daha dengeli çıktılar üretir, çünkü ortalaması sıfıra yakındır.
Bu da özellikle negatif/pozitif verilerle çalışırken avantaj sağlar.
Formül:
f(x)=ex−e−xex+e−xf(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}f(x)=ex+e−xex−e−x 
1. Classification (Sınıflandırma):
Görüntünün tamamına bakarak tek bir sınıf etiketi atama işlemidir.
Örn: “Bu bir kedi mi, köpek mi, araba mı?” gibi soruları yanıtlar.
2. Detection (Nesne Tespiti)
Bir görüntüde birden fazla nesne varsa, bunları etiketleyerek konumlarını (kutu içinde) tespit eder.
Nerede ve ne var?” sorusunu yanıtlar.
3. Segmentation (Görüntü Segmentasyonu)
Görüntüdeki her bir pikselin hangi nesneye ait olduğunu belirler.
En ayrıntılı görevdir. "Ne var ve tam olarak neresinde?"
