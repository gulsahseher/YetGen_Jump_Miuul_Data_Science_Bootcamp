#######################################################
# FONKSİYINLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
#######################################################
# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehesions

#######################################################
# FONKSİYONLAR (FUNCTIONS)
#######################################################

##########################
# Fonksiyon Okuryazarlığı
##########################

print("a", "b")

# Argümanlar bir özellik belirtebileceği gibi fonksiyonun genel amacını
# biçimlendirmek için kullanılan alt görevcilerdir.

print("a", "b" , sep="__")

##########################
# Fonksiyon Tanımlama
##########################


def calculate(x):
    x=2*x
    return x

calculate(4)

# İki argümanlı/parametreli bir fonksiyon tanımlayalım.

def summer(arg1, arg2):
    print(arg1+arg2)

summer(41, 34)

summer(arg2=41, arg1=34)

##########################
# Docstring
##########################

# Fonksiyonlara herkesin anlayabileceği ortak bir dil ile bilgi notu ekleme yoludur.
# Docstring açmak 3 çift tırnak kullanılır.

def summer(arg1, arg2):
    print(arg1+arg2)

def summer(arg1, arg2):
    """
    Sum of two numbers

    Args:
        arg1: int, float
        arg2: int,float

    Returns: int,float

    Examples:

    Notes:

    """
    print(arg1+arg2)

summer(1,3)

##########################
# Fonksiyonların Statement/Body Bölümü
##########################

# def function_name(parameters/arguments):
#     statements (function body)

def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")

say_hi()

def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")

say_hi("miuul")

def multiplication(a, b):
    c = a * b
    print(c)

multiplication(10, 9)

# girilen değerleri bir liste içinde saklayacak fonksiyon
# Bütün çalışmalar içerisinde erişilebilen değerlere global etki alanındaki değişkenler denir.
# İlgili fonksiyon döngü, if yapıları içerisinde oluşturulan ve bu döngü alanları içerisinde kalan değişkenlere local değişkenler denir.


list_store=[]

def add_element(a,b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(1, 8)
add_element(18, 8)
add_element(180, 10)

##########################
# Ön Tanımlı Argümanlar/Parametreler (Default Parameters/Arguments)
##########################

def divide(a, b):
    print(a / b)

divide(1, 2)

def divide(a, b=1):
    print(a / b)

divide(10)

def say_hi(string="Merhaba"):
    print(string)
    print("Hi")
    print("Hello")

say_hi("mrb")
say_hi()

##########################
# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?
##########################

# Birbirini tekrar eden değişen durumlar varsa bunları bir fonksiyonca tanımlayıp
# bu fonksiyonu otomatik bir şekilde çağırıp daha kolay bir şekilde kullanma işlemi gerçekleştirilebilir.

# varm, moisture, charge

(56 + 15 ) / 80
(17 + 45 ) / 70
(52 + 45 ) / 80

# DRY

def calculate(varm, moisture, charge):
    print((varm + moisture)/charge)

calculate(56, 15, 80)

##########################
# Return: Fonksiyon Çıktılarını Girdi Olarak Kullanmak
##########################

# Fonksiyon çıktısını kullanmak için return kullanılır.

def calculate(varm, moisture, charge):
    print((varm + moisture)/charge)

# calculate(56, 15, 80) * 10

def calculate(varm, moisture, charge):
    return (varm + moisture)/charge

a = calculate(56, 15, 80) * 10

def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return varm, moisture, charge, output

type(calculate(56, 15, 80) * 10)

varm, moisture, charge, output = calculate(56, 15, 80)

##########################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
##########################

def calculate(varm, moisture, charge):
    return int((varm + moisture)/charge)

calculate(56, 15, 80) * 10

def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(45, 1)

def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculation(1, 3, 5, 12)

def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)

all_calculation(1, 3, 5, 19, 12)

##########################
# Lokal & Global Değişkenler (Local & Global Variables)
##########################

# local etki alanından global etki alanına değiştirilebilir

list_store = [1, 2] # global etki alanı
type(list_store)

def add_element(a, b):
    c = a * b # local etki alanı
    list_store.append(c)
    print(list_store)

add_element(1, 9)

#######################################################
# KOŞULLAR (CONDITIONS)
#######################################################

# True-False'u hatırlayalım.
1==1
1==2

##########################
# If
##########################
if 1==1:
    print("something")

if 1==2:
    print("something")

number = 11

if number == 10:
    print("number is 10")

number = 10
number  = 20

def number_check(number):
    if number == 10:
        print("number is 10")

number_check(10)

##########################
# Else
##########################

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)

##########################
# Elif
##########################

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")

number_check(9)

#######################################################
# DÖNGÜLER (LOOPS)
#######################################################

##########################
# for loop
##########################

# Üzerinde iterasyon yapılan nesneler üzerinde gezinmeyi ve
# bu gezinmeden sonucunda yakalanacak her bir eleman üzerind eçeşitli işlemler yapılmasını sağlar

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student)

for student in students:
    print(student.lower())

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary*20/100+salary))

for salary in salaries:
    print(int(salary*30/100+salary))

for salary in salaries:
    print(int(salary*50/100+salary))

def new_salary(salary, rate):
    return int(salary*rate/100+salary)

new_salary(1500, 10)
new_salary(2000, 20)

for salary in salaries:
    print(new_salary(salary, 10))

salaries2 = [10700, 25000, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))

##########################
# Uygulama - Mülakat Sorusu
##########################

# Amaç: Aşağıdaki şekilde string değiştiren  fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def alternating(string):
    new_string = ""
    # girilen stringlerin index'lerinde gez.
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            new_string = new_string + string[string_index].upper()
        # index tek ise küçük harfe çevir
        else:
            new_string = new_string + string[string_index].lower()

    return new_string

alternating("hi my name is john and i am learning python")

##########################
# break & continue & while
##########################

salaries = [1000, 2000, 3000, 4000, 5000]

# koşula geldiğinde döngüyü durdur
for salary in salaries:
    if salary == 3000:
        break
    print(salary)

# koşula geldiğinde devam et, çalışma, diğer iterasyona geç
for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

# while True olduğu sürece döngüye devam et
number = 1
while number < 5:
    print(number)
    number += 1

##########################
# Enumerate: Otomatik Counter/Indexer ile for loop
##########################

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

##########################
# Uygulama - Mülakat Sorusu
##########################
# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun.

A = []
B = []
C = []

students = ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

divide_students(students)

##########################
# Alternating Fonksiyonunun Enumerate ile Yazılması
##########################

def alternating_with_enumerate(string):
    new_string = ""
    # girilen stringlerin index'lerinde gez.
    for i, letter in enumerate(string):
        # index çift ise büyük harfe çevir
        if i % 2 == 0:
            new_string = new_string + letter.upper()
        # index tek ise küçük harfe çevir
        else:
            new_string = new_string + letter.lower()

    return new_string

alternating_with_enumerate("hi my name is john and i am learning python")

##########################
# Zip
##########################

# Ayrı listeleri tek bir liste içerisine her birisinde bulunan elemanları aynı sırada
# bir araya getirerek her birisini görebileceğimiz formda bir liste haline getirir.

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))

#######################################################
# lambda, map, filter, reduce
#######################################################

##########################
# Lambda
##########################
# Değişkenler bölümünde yer tutmadan ihtiyaç duyulan bir bölümde kullanılıp atılması anlamına gelir.

def summer(a, b):
    return a + b

summer(1,3) + 9

new_sum = lambda a, b: a + b

new_sum(4, 5)

##########################
# Map
##########################
# Bir fonksiyon ve bu fonksiyonun uygulanmak istendiği iteratif bir nesne verilmesi gerekmektedir.
# Bütün elemanlara döngü yazılmış gibi uygulanmasını sağlar

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

# del new_sum

list(map(lambda x: x * 20 / 100 + x, salaries))

list(map(lambda x: x ** 2 / 100 + x, salaries))

##########################
# Filter
##########################
# Sorgu noktası gibi düşünülebilir.

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

##########################
# Reduce
##########################
# İteratif bir şekilde ilgili elemanlara belirli işlemlerin uygulanmasını sağlar

from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)

#######################################################
# Comprehesions
#######################################################

##########################
# List Comprehension
##########################

# Örneğin elimizde bir maaş listesi olsun ve bu maaşlara %20 zam yapılmasını isteyelim.
# Önce bu işlemleri normal döngüler ve koşullar yöntemi  ile yapalım

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary*2))

# List comprehensions yöntemi ile bu işlemler tekrardan yapalım.

[salary * 2 for salary in salaries] # [2000, 4000, 6000, 8000, 10000]

[salary * 2 for salary in salaries if salary < 3000] # [2000, 4000]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries] # [2000, 4000, 0, 0, 0]

[new_salary(salary * 2) * 2 if salary < 3000 else new_salary(salary * 0.2) for salary in salaries] # [4800.0, 9600.0, 720.0, 960.0, 1200.0]
                                                                                                   # if - else yapısı kullanılacaksa for döngüsü if - else'in sağında yer alır
# Senaryo: Bir listede tüm öğrenciler yer alırken diğer listede istemediğimiz öğrenciler yer almaktadır.
# students_no listesindeki öğrencilerin isimleri küçük ve bu liste dışında kalan öğrencilerin isimleri büyük yazılsın.

students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.upper() if student in students_no else student.lower() for student in students] # ['JOHN', 'mark', 'VENESSA', 'mariam']

[student.lower() if student not in students_no else student.upper() for student in students] # ['JOHN', 'mark', 'VENESSA', 'mariam']

##########################
# Dict Comprehension
##########################

dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v**2 for (k,v) in dictionary.items()} # {'a': 1, 'b': 4, 'c': 9, 'd': 16}

{k.upper(): v**2 for (k,v) in dictionary.items()} # {'A': 1, 'B': 4, 'C': 9, 'D': 16}

##########################
# Uygulama - Mülakat Sorusu
##########################

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir.
# Key'ler orijinal değerler value'lar ise değiştirilmiş değerler olsun.

numbers = range(10)
new_dict={}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n**2 for n in numbers if n % 2 == 0}

##########################
# List & Dict Comprehension Uygulamalar
##########################

# Uygulama 1
# Bir veri setindeki değişken isimlerini değiştirmek

# before:
# ["total", "speeding", "alcohol", "not_distracted", "no_previous", "ins_premium", "ins_losses", "abbrev"]

# after:
# ["TOTAL", "SPPEDING", "ALCOHOL", "NOT_DISTRACTED", "NO_PREVIOUS", "INS_PREMIUM", "INS_LOSSES", "ABBREV"]

import seaborn as sns

df = sns.load_dataset("car_crashes")

df.columns

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

print(A)

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns] #Index(['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS','INS_PREMIUM', 'INS_LOSSES', 'ABBREV'],dtype='object')

# Uygulama 2
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.

# before:
# ["TOTAL",
#  "SPPEDING",
#  "ALCOHOL",
#  "NOT_DISTRACTED",
#  "NO_PREVIOUS",
#  "INS_PREMIUM",
#  "INS_LOSSES",
#  "ABBREV"]

# after:
# ["NO_FLAG_TOTAL",
#  "NO_FLAG_SPPEDING",
#  "NO_FLAG_ALCOHOL",
#  "NO_FLAG_NOT_DISTRACTED",
#  "NO_FLAG_NO_PREVIOUS",
#  "FLAG_INS_PREMIUM",
#  "FLAG_INS_LOSSES",
#  "NO_FLAG_ABBREV"]

["FLAG_"+ col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_"+ col if "INS" in col else "NO_FLAG_" + col for col in df.columns] # değişiklikleri kalıcı hale getirebilmek için yapılır.

# Uygulama 3
# Amaç: Key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak
# Bu işlemi sadece sayısal değişkenler için yap

# Output:
# {"total": ["mean", "min", "max", "var"],
#  "speeding": ["mean", "min", "max", "var"],
#  "alcohol": ["mean", "min", "max", "var"],
#  "not_distracted": ["mean", "min", "max", "var"],
#  "no_previous": ["mean", "min", "max", "var"],
#  "ins_premium": ["mean", "min", "max", "var"],
#  "ins_losses": ["mean", "min", "max", "var"]}

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"] # Kategorik olmayan yani object olmayan değişkenleri getirir. Nümerik tipteki değişkenleri getirir.

# Döngü yöntemi
dict = {}
agg_list = ["mean", "min", "max", "sum"]

for col in df.columns:
    dict[col] = agg_list

dict

# Dict Comprehensions yöntemi
{col: agg_list for col in num_cols}

new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict) # istenen bir fonksiyon setinin değişkenlere uygulanarak sonuç elde edilmesini sağlar