# #####################################################
# Python Alıştırmalar
# #####################################################

################
# Görev 1
################

x = 8
type(x) #int

y = 3.2
type(y) #float

z = 8j + 18
type(z) #complex

a = "Hello World"
type(a) #string

b = True
type(b) #bool

c = 23 < 22
type(c) #bool

l = [1, 2, 3, 4]
type(l) #list

d = {"Name": "Jake",
     "Age": 27,
     "Adress" : "Downtown"}
type(d) #dict

t = ("Machine Learning", "Data Science")
type(t) #tuple

s = {"Python", "Machine Learning", "Data Science"}
type(s) #set


################
# Görev 2
################

# Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz.
# Kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into insight."
text = text.upper()
text = text.strip(".")
text = text.replace(",","")
text = text.split()
print(text)

text.upper().replace(",", "").replace(".","").split()
################
# Görev 3
################

# Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Verilen listenin eleman sayısına bakınız.

len(lst)

# Sıfırıncı ve onuncu indeksteki elemanları çağırınız.

lst[0]
lst[10]

# Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.

lst[0:4]

# Sekizinci indeksteki elemanı siliniz.

lst.pop(8)

# Yeni bir eleman ekleyiniz.

lst.append(":)")
lst

#Sekizinci indekse "N" elemanını tekrardan ekleyiniz.

lst.insert(8, "N") #indexe göre değer atama
lst

################
# Görev 4
################

dict = {"Christian":["America",18],
        "Daisy":["England",12],
        "Antonio":["Spain",22],
        "Dante":["Italy",25]}

# Key değerlerine erişiniz.

dict.keys()

# Value'lara erişiniz.

dict.values()

# Daisy key'ine 12 değerini 13 olarak güncelleyiniz.

dict["Daisy"][1] = 13
dict

# Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.

dict.update({"Ahmet": ["Turkey", 24]})
dict

# Antonio'yu dictionary'den siliniz.

dict.pop("Antonio")
dict

################
# Görev 5
################

# Argüman olarak bir liste alsın.
# Listenin içerisindeki tek ve çift sayıları ayrı listere atasın.
# Bu listeleri return etsin

l = [2, 13, 18, 93, 22]

def func(li):
    even_list = []
    odd_list = []
    for i in range(len(li)):
        if li[i] % 2== 0:
            even_list.append(li[i])
        else:
            odd_list.append(li[i])

    return even_list, odd_list

even_list, odd_list = func(l)
print(even_list)
print(odd_list)

#2. yol
l = [2, 13, 18, 93, 22]

def func(li):
    even_list = []
    odd_list = []
    for i in li:
        if i % 2== 0:
            even_list.append(i)
        else:
            odd_list.append(i)

    return even_list, odd_list

even_list, odd_list = func(l)
print(even_list)
print(odd_list)

################
# Görev 6
################

# Verilen listede mühendislik ve tıp fakültelerinde dereceye giren öğrencilerin isimleri vardır.
# Sıırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil eder.
# Son üç öğrenci tıp fakültesi öğrenci sırasına aittir.
# Enumerate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for student_index, students in enumerate(ogrenciler):
    if student_index < 3:
        print(f"Mühendislik Fakültesi {student_index + 1}. öğrenci: {students}")
    else:
        print(f"Tıp Fakültesi {student_index - 2}. öğrenci: {students}")

################
# Görev 7
################

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30, 75, 150, 25]

ders_bilgisi = list(zip(ders_kodu,kredi,kontenjan))
print(ders_bilgisi)

for i in range(len(ders_bilgisi)):
    print(f"Kredisi {ders_bilgisi[i][1]} olan {ders_bilgisi[i][0]} kodlu dersin kontenjanı {ders_bilgisi[i][2]} kişidir.")

# 2. yol
for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")


################
# Görev 8
################

# Eğer 1. küme 2. kümeyi kapsıyor ise ortak elemanlarını yazdır.
# Eğer 1. küme 2. kümeyi kapsamıyor ise 2. kümenin 1. kümeden farkını yazdır.

kume1 = set(["data", "python"])
kume2 = set(["data","function", "qcut", "lambda", "python", "miuul"])

def kume(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

kume(kume1, kume2)