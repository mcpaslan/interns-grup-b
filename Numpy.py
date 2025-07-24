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