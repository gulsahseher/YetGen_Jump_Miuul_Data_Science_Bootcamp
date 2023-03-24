############################################################
# VERİ YAPILARI (DATA STRUCTURES)
############################################################
# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set

############################################################
# Veri Yapılarına Giriş ve Hızlı Özet
############################################################

# Sayılar : integer
x = 46
type(x)

# Sayılar : float
x = 10.3
type(x)

# Sayılar : complex
x = 2j+1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)
5 == 4
1 == 1
type(3==2)

# Liste (list)
x = ["btc", "eth", "xrp"]
type(x)

# Sözlük (dictionary)
x = {"name":"Peter","Age":36} # {key:value}
type(x)

# Tuple
x = ("python","ml","ds")
type(x)

# Set
x = {"python","ml","ds"}
type(x)

# Not: Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.

############################################################
# Sayılar (Numbers): int, float, complex
############################################################

a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2

#####################
# Tipleri Değiştirme
#####################

int(b)
float(a)
int(a * b / 10)

############################################################
# Karakter Dizileri (Strings): str
############################################################

print("John")

"John" # ekrana basmak için print kullan

name = "John"

################################
# Çok Satırlı Karakter Dizileri
################################

"""
Veri Yapılarına Giriş ve Hızlı Özet
Sayılar (Numbers): int, float, complex
Karakter Dizileri (Strings): str
Boolean (TRUE-FALSE): bool
"""

long_str = """
Veri Yapılarına Giriş ve Hızlı Özet
Sayılar (Numbers): int, float, complex
Karakter Dizileri (Strings): str
Boolean (TRUE-FALSE): bool
"""

################################
# Karakter Dizilerinin Elemanlarına Erişmek
################################

name
name[0]
name[3]
name[0:2] # Karakter dizinlerinde slice işlemi
long_str[0:10]

################################
# String İçerisinde Karakter Sorgulama
################################

long_str

"veri" in long_str

"bool" in long_str

################################
# String (Karakter Dizisi) Metodları
################################

dir(str)

type(name)
type(len)

# len : Stringlerin kaç elemandan oluştuğunu verir
len(name) #fonksiyon

# upper() & lower() : küçük-büyük dönüşümleri
"miuul".upper() #metod
'MIUUL'.lower() #metod

type(upper)

# Not: Metodlar class yapıları içerinde tanımlanır.

#replace : karakter değiştirme
hi = "Hello AI Era"
hi.replace("l","p")

#split : bölme
"Hello AI Era".split()

#strip : kırpma
" ofofo ".strip()
"ofofo".strip("o")

#capitalize: ilk harfi büyütür
"foo".capitalize()

dir("foo")
"foo".startswith("f")

############################################################
# Liste (List)
############################################################

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v","d"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]] #Listeler kapsayıcıdır.
not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]
not_nam[0:4]
type(not_nam[6])

notes[0]
notes[0] = 99

################################
# Liste Metodları (List Methods)
################################

dir(notes)

# len: builtin python fonksiyonu, boyut bilgisi.
len(notes)
len(not_nam)

# append: eleman ekler
notes.append(100)
notes

# pop: indexe göre eleman siler
notes.pop(0)
notes

# insert: indexe ekler
notes.insert(2, 99)
notes

############################################################
# Sözlük (Dictionary)
############################################################

# - Değiştirilebilir
# - Sırasızdır. (3.7 sonra sıralı.)
# - Kapsayıcıdır.

# key-value

dictionary = {"REG":"Regression",
              "LOG":"Logistic Regression",
              "CART":"Classification and Reg"}

dictionary["REG"]

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE",30]
              }

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30
              }

dictionary["REG"]
dictionary["CART"][1]


# Key Sorgulama
"REG" in dictionary
"YSA" in dictionary

# Key'e göre value'ya erişme
dictionary["REG"]
dictionary.get("REG")

# Value Değiştirme
dictionary["REG"] = ["YSA",10]

# Tüm Key'lere Erişmek
dictionary.keys()

# Tüm Value'lara Erişmek
dictionary.values()

# Tüm Çiftleri Tuple Halinde Listeye Çevirme
dictionary.items()

# Key-Value Değerini Güncellemek
dictionary.update({"REG": 11})

# Yeni Key-Value Eklemek
dictionary.update({"RF": 10})

############################################################
# Demet (Tuple)
############################################################

# - Değiştirilemez.
# - Sırasılıdır.
# - Kapsayıcıdır.
# - Kullanımı düşüktür

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99 # Tuple olduğu için değiştirilemez bu yüzden hata verir.
          # Değiştirmek için liste formuna getirmek gerekir.

t = list(t)
t[0] = 99
t = tuple(t)

############################################################
# Set
############################################################

# - Değiştirilebilir.
# - Sırasızdır + Eşsizdir.
# - Kapsayıcıdır.

# difference(): İki kümenin farkı
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.difference(set2) # set1'de olup set2'de olmayanlar
set2.difference(set1)
set1-set2

# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
set1.symmetric_difference(set2)

# intersection(): İki kümenin kesişimi
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)
set2.intersection(set1)
set1 & set2 # ve iki kümenin kesişimi

# union(): İki kümenin birleşimi
set1.union(set2)
set2.union(set1)

# isdisjoint(): İki kümenin kesişimi boş mu?
set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

# issubset(): Bir küme diğer kümenin alt kümesi mi?
set1.issubset(set2)
set2.issubset(set1)

# issuperset(): Bir küme diğer kümeyi kapsıyor mu?
set2.issuperset(set1)
set1.issuperset(set2)

