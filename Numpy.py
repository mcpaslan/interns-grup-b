import numpy as np

a = np.array([1,2,3,4,])
print(a, "\n")

skaler = np.array(1)
print(skaler, "\n")

vektorel = np.array([1,2,3,4,5,6])
print(vektorel, "\n")

matris = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matris, "\n")

tensor = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
print(tensor, "\n")

#Skaler (0D)

#Vektör (1D)

#Matris (2D)

#Tensor (3D)

'''NumPy (Numerical Python), hızlı, verimli ve kolay matematiksel işlemler yapmamıza olanak sağlar. 
Özellikle bilimsel hesaplamalar, veri analizi ve makine öğrenmesi için çok kullanışlıdır.'''

a = np.array([1,2,3])
b = np.array([4,5,6])
print(a + b)

import time

z1 = time.time()
liste=np.arange(1_000_000)**100

z2 = time.time()

print(z2 - z1)

vektor = np.array([1,2,4,8,16,32])
print(vektor[0])
print(vektor[-1])
print(vektor[2:4])
print(vektor[:4])
print(vektor[1:4:2])

vektor = np.array([1,2,4,8,16,32])
matris = np.array([[1,2,4],[8,16,32],[64,128,256]])

print(vektor)
print(matris)
print(vektor.size)
print(matris.size)
print(vektor.shape)
print(matris.shape)
print(vektor.ndim)
print(matris.ndim)
print(vektor.dtype)
print(matris.dtype)
print(type(vektor))
print(type(matris))

sayilar = np.array([1,2,4,8,16])
print(sayilar.dtype)

sayilar2 = np.array([1,2,4,8.0,16])
print(sayilar2.dtype)

sayilar3 = np.array([1,2,4,8.0,"ufuk"])
print(sayilar3.dtype)

sayilar2 = sayilar.astype("int64")
print(sayilar2.dtype)

tarih = np.array(["2013-01-01"])
print(tarih.dtype)

tarih = tarih.astype("datetime64")
print(tarih.dtype)

matris1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(matris1)
print(matris1[0])
print(matris1[0,:2])
print(matris1[0:4,0:1])
print(matris1[0:4,1:3])
print(matris1[1:3,1:3])

dir(np.random)
print("hello")
print(np.random.randint(0,5))
print(np.random.randint(0,5,(4,4)))

range(1000000)

range (0,1000000)
million = np.arange(1000000)
print(million)
print(np.arange(1,20,4))
print(np.linspace(0,18,9))

l1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(l1.shape)
print(l1.size)
print(l1.reshape(4,3))
print(l1.reshape(12))

np.random.randint(0,2,(8,6)).reshape(4,12)
print(np.random.randint(0,2,(8,6)).reshape(4,12))
print(np.linspace(0,2,30).reshape(6,5))

matris2 = np.random.randint(0,5,(4,4))
print(matris2)

matris3  = matris2.view()
print(matris3)

matris3[0] = [9,9,9,9]
print(matris3)

print(matris2)

matris4 = np.random.randint(0,5,(4,4))
print(matris4)

matris5 = matris4.copy()
print(matris5)

matris5[0] = [9,9,9,9]
print(matris5)

print(np.zeros((5,4)))
print(np.zeros(5))
print(np.ones((3,3)))
print(np.ones(3))
print(np.full((3,3),5))
print(np.identity(4))
print(np.eye(4,4))
print(np.eye(4,k=1))
print(np.diag([1,2,3,2,1]))
print(np.diag([1,2,3]))
x= np.random.randint(0,2,(4,4))
print(x)

print(np.diag(x))
m1=np.ones((5,5))
m2=np.full((5,5),8)
m1 = m1.astype("int64")

print(m2)
print(m1*4)
print(m2 + 3)
print(m1 + m2)
print(m1/2)
print(m2/3)
print(m1.dtype)
print(m1//3)

m3 = np.ones((5,4))
m4 = np.full((5,5),5)
print(m3)
print(m4)

m3 = m3.reshape(20)
m4 = m4.reshape(25)[:20]
print(m3)
print(m4)
m3 = m3.reshape(5,4)
m4 = m4.reshape(5,4)
print(m3 + m4)

m5 = np.random.randint(0,5,(5,5))
print(m5)
print(m5>2)
print(m5[m5>2])
print(m5%2==0)
print(m5[m5%2==0])
sifir=m5!=0
cift=m5%2==0
print(sifir)
print(cift)
print(m5[cift & sifir])