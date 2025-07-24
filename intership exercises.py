print("hello world")
print("Ufuk'un evi")
print('Ufuk\'un Evi')
print("""Ufuk'un
Evi""")
print("Ufuk'un\nEvi")
print("Ufuk'un\tEvi")

Ad = "Ufuk"
Soyad = "Vural"

print(Ad + " " + Soyad)
print(Ad[0] + Soyad[0])
print(Ad[0]+Ad[1])
print(Ad[-1])
print(Ad[0:4])
print(Ad[::])
print(Ad[::2])
print(Ad[::-1])
print(Ad.upper())
print(Ad.lower())
# Ad = Ad.upper()
# print(Ad)
# Ad = Ad.lower()
# print(Ad)

print(Ad.startswith("Uf"))
print(Ad.endswith("k"))
print(len(Ad))
print(len(Ad+Soyad))
print(Ad *10)
Yas = 20

print("{} , {} yaşındadır".format(Ad,Yas))
mesaj = "merhaba"
print("{} , {} dedi".format(Ad,mesaj))
print(f"{Ad} {mesaj} dedi")

sayi1 = 5
sayi2 = 3.8
sayi3 = 4/5

print(type(sayi1))
print(type(sayi2))
print(type(sayi3))

sayi4 = 4**2
sayi5 = 4*2

print(sayi4)
print(sayi5)

#toplama "+"
#çıkarma "-"
#çarpma "*"
#bölme "/"
#kuvvet alma "**"
#tamsayı bölmesi "//"
#mutlak değer "abs"
#yuvarlama "round"

sayi6 = 3.56789
print(round(sayi6))
print(round(sayi6,3))

#Karşılaştırma Operatörleri

#Eşittir = "=="
#Küçüktür = "<"
#Büyüktür = ">"
#Küçük eşittir = "<="
#Büyük eşittir = ">="
#Eşit değildir = "!"

sayi7 = 5

print(sayi7==4)
print(sayi7==5)

sayi8 = "100"
sayi9 = 100
sayi10 = int(sayi8)

print(sayi10 == sayi9)

sayi11 = 123

sayi12= str(sayi11)

print(sayi12)

sayi13 = 5
sayi13 *= 3
print(sayi13)

renkler = ["yesil" , "mavi" , "mor" , "sari" , "turuncu"]
print(type(renkler))
print(renkler)
print(len(renkler))
print(renkler[1])
print(renkler[1:3])

renkler.append("gri")
print(renkler)
renkler.insert(0,"beyaz")
print(renkler)
renkler.remove("sari")
print(renkler)

renkler2 = ["kirmizi","siyah"]
renkler.extend(renkler2)
print(renkler)
renkler.pop()
print(renkler)
renkler.reverse()
print(renkler)

sayilar =["1", "2", "7", "9", "13"]
print(max(sayilar))
sayilar2 = [1,2,7,9,13]
print(max(sayilar2))
print(sum(sayilar2))

for i in renkler:
    print(i)

print(list(enumerate(renkler)))
stringrenkler = "-".join(renkler)
print(stringrenkler)

renkler2 = stringrenkler.split("-")
print(renkler2)

#tuple fonksiyonları
tuple = ("Mavi", "Sari", "Kirmizi", "Mor")
print(len(tuple))
print(type(tuple))

for i in tuple:
    print(i)

set = {"mavi","sari","kirmizi","mor"}
print(type(set))

for i in set:
    print(i)

set.add("pembe")
print(set)
set.discard("mavi")
print(set)

set1 = {"mavi","mor","pembe"}
set2 = {"sari","kirmizi","mor"}
print(set1.intersection(set2))
print(set1.union(set2))
print(set1.difference(set2))
print(set2.difference(set1))

#boş küme,liste,tuple oluşturmak

#bosliste1 = []
#bosliste2 = list()

#bostuple1 = ()
#bostuple2 = tuple()

#boskume = set()

sayi = 0
while sayi < 5:
    print("Sayım:", sayi)
    sayi += 1

ogrenci = {
    "ad": "Ufuk",
    "soyad": "Vural",
    "yas": 20,
    "notlar": [90, 85, 100]
}

print(ogrenci["ad"])
print(ogrenci["notlar"])

ogrenci["okul"] = "Mskü"
print(ogrenci)

ogrenci["yas"] = 20
print(ogrenci)

for key in ogrenci:
    print(key, ":", ogrenci[key])

for anahtar, deger in ogrenci.items():
    print(f"{anahtar} -> {deger}")

kisi = {"isim": "ufuk", "yas": 20,"cinsiyet": "erkek", }
print(kisi)
print(kisi["isim"])

kisi.update({"isim": "Vural", "yas": 21})
print(kisi)

kisi["id"] = 12345
print(kisi)

del kisi["id"]
print(kisi)

for x in kisi:
    print(x)

print(kisi.keys())
print(kisi.values())
print(kisi.items())

for k,v in kisi.items():
    print(k,v)