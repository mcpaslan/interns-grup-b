# CNN ile Kedi ve Köpek sınıflandırılması

Bu proje, **Convolutional Neural Network (CNN)** kullanılarak kedi ve köpek görsellerinin sınıflandırılmasını amaçlar. 
Görseller, **Kaggle** üzerinden sağlanan [Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) kullanılarak eğitilmiştir.

## Veri Seti

- Veri seti Kaggle'dan indirilmiştir: [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)
- Veri seti içerisinde 25.000 görsel bulunmaktadır:
  - `cat.0.jpg` → `cat.12499.jpg`
  - `dog.0.jpg` → `dog.12499.jpg`

> Görseller eğitim sürecinde `cats/` ve `dogs/` klasörleri altında organize edilmiştir.

---

## Kullanılan Model: CNN

Model, temel CNN mimarisi kullanılarak TensorFlow ile eğitilmiştir:

Not: TensorFlow, genellikle Python 3.7 - 3.11 sürümleriyle uyumludur. Daha eski veya daha yeni sürümlerde kurulumda sorunlar yaşanabilir.

