##################################################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
##################################################################
# - Numpy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

##################################################################
# NUMPY
##################################################################

# Neden NumPy? (Why NumPy?)
# Numpy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attributes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler
# Matematiksel İşlemler (Mathematical Operations)

#####################################
# Neden NumPy? (Why NumPy?)
#####################################

# - Numpy'ın listelerden farklılaştığı nokta verimli veri saklama ve yüksek seviyeden işlemlerdir(vektörel işlemlerdir).
# - Numpy içerisinde veri tutarken fixtype adı verilen sabitlenmiş veri tipinde tutarak listelere
# göre çok daha hızlı bir şekilde işlem yapma imkanı sunar.
# - Döngü yazmaya gerek olmadan Array seviyesinde çok basit işlemlerle normalde çok daha fazla çaba gerektiren işlemlerin
# yerine getirilmesini sağlar.

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab=[]

for i in range(0, len(a)):
    ab.append(a[i]*b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b #vektörel işlemler

#####################################
# Numpy Array'i Oluşturmak (Creating Numpy Arrays)
#####################################
import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)

np.random.randint(0, 10, size=10) # Hangi sınırdan hangi sınıra kadar array oluşturmak istiyorsun?

np.random.normal(10, 4, (3, 4)) # normal dağılımlı

#####################################
# NumPy Array Özellikleri (Attributes of Numpy Arrays)
#####################################
import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: aray veri tipi

a = np.random.randint(10, size=5)

a.ndim # 1
a.shape # (5,) tek boyutlu 5 elemanlı
a.size # 5
a.dtype # int32

#####################################
# Yeniden Şekillendirme (Reshaping)
#####################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

#####################################
# Index Seçimi (Index Selection)
#####################################
import numpy as np

a = np.random.randint(10, size=10)
a[0]
a[0:5] #slicing
a[0] = 999

m = np.random.randint(10, size = (3,5))
m

m[0, 0]
m[1, 1]
m[2, 3]

m[2, 3] = 999
m

m[2, 3] = 2.9
m

m[:,0]
m[1,:]
m[0:2,0:3]

#####################################
# Fancy Index
#####################################
import numpy as np

v = np.arange(0, 30, 3) # belirli bir adım boyunca array oluşturma işlemi

v[1]
v[4]

catch = [1,2,3] # index bilgisi ilgili array bilgisiyle seçim işlemi gerçekleşir
v[catch]

#####################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#####################################
# İlgili array ifadesinin içerisine koşul ifadesini yerleştir.

import numpy as np

v = np.array([1, 2, 3, 4, 5])

# Klasik bir döngü ile

ab = []

for i in v:
    if i < 3:
        ab.append(i)

# Numpy ile

v < 3 # array([ True,  True, False, False, False])

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v <= 3]

#####################################
# Matematiksel İşlemler (Mathematical Operations)
#####################################

import numpy as np

v = np.array([1, 2, 3, 4, 5])

v / 5 #array([0.2, 0.4, 0.6, 0.8, 1. ])
v * 5 / 10 #array([0.5, 1. , 1.5, 2. , 2.5])
v ** 2 #array([ 1,  4,  9, 16, 25])
v - 1 #array([0, 1, 2, 3, 4])

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v) #15
np.min(v)
np.max(v)
np.var(v) #2.0

# Numpy'da İki Bilinmeyenli Denklem Çözümü

5*x0 + x1 = 12
x0 + 3*x1 = 10

a = np.array([[5,1], [1,3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

##################################################################
# PANDAS
##################################################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri

#####################################
# Pandas Series
#####################################

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

#####################################
# Veri Okuma (Reading Data)
#####################################

import pandas as pd

df = pd.read_csv("C:/Users\GULSAH\Documents\PythonScripts\Arcelik\Doluluk_Oranı/test.csv")
df.head()

#####################################
# Veriye Hızlı Bakış (Quick Look at Data)
#####################################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info() #bilgi
df.columns #df'in değişkenlerinin ismi
df.index #index bilgisi
df.describe().T #df'in özet istatistik bilgileri
df.isnull().values.any() #eksik değer var mı diye sorar
df.isnull().sum() #hangi değişkende kaç tane eksik değer olduğunu bulmamızı sağlar
df["sex"].head()
df["sex"].value_counts() #kategorik değişkenin sınıfları ve bu sınıfların kaçar tane olduğunu verir

#####################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#####################################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]

df.drop(0, axis=0).head() #satır veya sütunu silme işlemi

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# silme işlemini kalıcı hale getirmek için
# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True) değişikliğin kalıcı olması gerektiğini veren bir argümandır

# Değişkeni indexe çevirmek

df["age"].head()
df.age.head()

df.index = df["age"]
df

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

# Indexi değişkene çevirmek

df.index

#1.yol
df["age"] = df.index
df.head()
df.drop("age", axis=1, inplace=True)

#2.yol
df.reset_index().head()
df = df.reset_index()
df.head()

# Değişkenler Üzerinde İşlemler

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None) # gösterilebilecek maksimum sütun sayısı olmasın
                                           # sütunların hepsini göstersin
df = sns.load_dataset("titanic")
df.head()

"age" in df
df["age"].head()
df.age.head()
type(df.age.head()) #pandas.core.series.Series

df[["age"]].head() #seçim işlemi yapılırken veri yapısının bozulmaması için [] içerisinde girilmelidir
type(df[["age"]].head()) #pandas.core.frame.DataFrame

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]
df

df.drop("age3",axis=1,).head()
df.drop(col_names ,axis=1,).head()

df.loc[:,~df.columns.str.contains("age")] #içerisinde age ifadesini barındıran değişkenleri siler
                                          #loc df'lerde seçme işlemleri için kullanılan bir yapıdır

# iloc & loc

# iloc: index bilgisi vererek seçim yapma işlemlerini ifade eder. (integer based selection)
# loc: mutlak olarak indexlerdeki labellara göre seçim işlemlerini yapar. (label based selection)

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection

df.iloc[0:3]
df.iloc[0,0]
df.iloc[0:3, 0:3]

# loc: label based selection

df.loc[0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names] #birden fazla değişken de isimleri ifade edilerek seçilebilir

# Koşullu Seçim (Conditional Selection)

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head() #yaşı 50'den büyük olan kişiler
df[df["age"] > 50]["age"].count() #yaşı 50'den büyük olan kaç kişi vardır
df.loc[df["age"] > 50, "class"] #yaşı 50'den büyük olan kişilerin sınıf bilgisi
df.loc[df["age"] > 50, ["age", "class"]].head() #yaşı 50'den büyük olan kişilerin sınıf ve yaş bilgisi
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head() #yaşı 50'den büyük olan ve erkek olan kişilerin sınıf ve yaş bilgisi
                                                                          #birden fazla koşul girilecekse parantez içine alınmalı
df.loc[(df["age"] > 50) &
       (df["sex"] == "male") &
       (df['embark_town'] == "Cherbourg"),
        ["age", "class", 'embark_town']].head() #yaşı 50'den büyük olan, erkek olan ve liman bilgisi Cherbourg olan kişilerin sınıf, yaş  ve şehir bilgisi

df['embark_town'].value_counts()
df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male") &
       ((df['embark_town'] == "Cherbourg") | (df['embark_town'] == "Southampton")),
        ["age", "class", 'embark_town']] #yaşı 50'den büyük olan, erkek olan ve liman bilgisi Cherbourg veya Southampton olan kişilerin sınıf, yaş  ve şehir bilgisi
df_new['embark_town'].value_counts()

#####################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
#####################################

#  Toplulaştırma özet istatistikler veren fonksiyonlardır
# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean() #cinsiyete göre yaşların ortalamasını alır
df.groupby("sex").agg({"age":["mean","sum"]}) #cinsiyete göre yaşların ortalamasını ve toplamını alır

df.groupby("sex").agg({"age":["mean","sum"],
                       "survived":"mean"}) #cinsiyete göre yaşların ortalamasını ve toplamını alır ve hayatta kalanların ortalamasını alır

df.groupby(["sex","embark_town"]).agg({"age":["mean","sum"],
                                       "survived":"mean"}) #cinsiyete ve limana göre yaşların ortalamasını ve toplamını alır ve hayatta kalanların ortalamasını alır

df.groupby(["sex","embark_town", "class"]).agg({"age":["mean","sum"],
                                       "survived":"mean"}) #cinsiyete, limana ve sınıfa göre yaşların ortalamasını ve toplamını alır ve hayatta kalanların ortalamasını alır

df.groupby(["sex","embark_town", "class"]).agg({
    "age":["mean","sum"],
    "survived":"mean",
    "sex":"count"}) #cinsiyete, limana ve sınıfa göre yaşların ortalamasını ve toplamını, hayatta kalanların ortalamasını ve cinsiyete göre hayatta kalanların sayısını alır

# Pivot Table

# Verisetini kırılımlar açısından değerlendirmek ve
# ilgilendiğimiz özet istatistiği bu kırılımlar açısından görme imkanını sağlar

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")
df.pivot_table("survived", "sex", "embarked", aggfunc= "std") #kesişimde değişkenlerin standart sapmaları hesaplanır.

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_age"] = pd.cut(df["age"],[0, 10, 18, 25, 40, 98])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option("display.width", 500) #df çıktısının sütunlarının bir kısmının alt tarafa inmeden yan yana görmemizi sağlar

##################################################################
# Apply ve Lambda
##################################################################

# Apply satır ya da sütunlarda otomatik olarak fonksiyon çalıştırmayı sağlar.
# Lambda bir fonksiyon tanımlama şeklidir. Kullan-at fonksiyonudur.

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

# Örneğin bütün age değişkenlerinin 10'a bölünmesini istiyoruz.

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

# Önce döngü ile yapalım

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10
        print(df[col].head())

df[["age","age2","age3"]].apply(lambda x: x/10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

# df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()
df.head()

##################################################################
# Birleştirme (Join) İşlemleri
##################################################################

# Birleştirme işlemleri genel olarak veritabanlarında yapılır. Python tarafına tekilleştirilmiş tablolar getirilir.

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

df1
df2

# Concat ile birlşetirme işlemi

pd.concat([df1, df2]) #alt alta birleştirmek için concat kullanılır
pd.concat([df1, df2], ignore_index=True) #indexleri sıfırlamak için

# Merge ile birleştirme işlemi

df1 = pd.DataFrame({"employees":["john", "dennis", "mark", "maria"],
                    "group":["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees":["mark", "john", "dennis", "maria"],
                    "start_date":[2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees") #hangi değişkene göre birleştirilmek istendiği belirtilmiştir

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                    "manager": ["Caner", "Mustafa","Berkcan"]})

pd.merge(df3, df4)

##################################################################
# Veri Görselleştirme: Matplotlib & Seaborn
##################################################################

#####################################
# Matplotlib
#####################################

# Kategorik değişken, sütun grafik ile görselleştirilir. (countplot bar)
# Sayısal değişken, histogram ve boxplot grafikleri ile görselleştirilir. (hist, boxplot)

#####################################
# Kategorik Değişken Görselleştirme
#####################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

#####################################
# Sayısal Değişken Görselleştirme
#####################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

# plt.hist(df["age"])
# plt.show()

plt.boxplot(df["fare"])
plt.show()

#####################################
# Matplotlib Özellikleri
#####################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# plot : Veriyi görselleştirmek için kullanılan fonksiyonlardan birisi

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y, "o")
plt.show()

# marker

import numpy as np
import matplotlib.pyplot as plt
y=np.array([13, 28, 11, 100])

plt.plot(y, marker = "o")
plt.show()
plt.interactive(False)

plt.plot(y, marker = "*")
plt.show()
plt.interactive(False)

markers = ["o", "*", ".", ",", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h"]

# line

y=np.array([13, 28, 11, 100])
plt.plot(y, linestyle = "dashdot", color="r")
plt.show()

# multiple lines

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

# label

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)

plt.title("Ana Başlık") # başlık
plt.xlabel("x ekseni") # x ekseni isimlendirmesi
plt.ylabel("y ekseni") # y ekseni isimlendirmesi

plt.grid() # ızgara ekleme
plt.show()

# Subplot

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()

#####################################
# Seaborn
#####################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")
df.head()

# Kategorik değişken görselleştirme

df["sex"].value_counts()
sns.countplot(x = df["sex"], data = df)
plt.show()

df["sex"].value_counts().plot(kind="bar")
plt.show()

# Sayısal değişkenleri görselleştirme

sns.boxplot(x = df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

df["total_bill"].plot.barh()
plt.show()

##################################################################
# Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)
##################################################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#####################################
# 1. Genel Resim
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print("############################# Shape #############################")
    print(dataframe.shape)
    print("\n")
    print("############################# Dtype #############################")
    print(dataframe.dtypes)
    print("\n")
    print("############################# Head #############################")
    print(dataframe.head(head))
    print("\n")
    print("############################# Tail #############################")
    print(dataframe.tail(head))
    print("\n")
    print("############################# NA #############################")
    print(dataframe.isnull().sum())
    print("\n")
    print("############################# Quantiles #############################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
    print("\n")

check_df(df)

df = sns.load_dataset("tips")
check_df(df)

df = sns.load_dataset("flights")
check_df(df)

#####################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

cat_summary(df, "sex")

# Bütün sütunlara (col) uygulaması için bir döngü yazdık

for col in cat_cols:
    cat_summary(df, col)

# cat_summary fonksiyonuna plot özelliğini ekleyelim

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

# Bool tipli değişkende hata verdiği için bool tipli değişkenlerin öncelikli olarak görselleştirmedik

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("Bool")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

# Bool tipli değişkenleri binary forma getirerek görselleştirdik

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# Döngüler genellikle fonksiyonun dışına yazılır. Bu yüzden sütunlarda gezmemizi sağlayan döngüyü cat_summary içine yazmadık.

# Tip sorgusunu da fonksiyon içerisinde yerine getirmek istersek

def cat_summary(dataframe, col_name, plot = False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#############################################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#############################################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "adult_male", plot=True)

# Hızlı analizi sağlayacak olan cat_summary fonksiyonunun ilk halidir.

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

cat_summary(df, "sex")

#####################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T


cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

# ######################################################################################################################
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numeric_col):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(dataframe[numeric_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

# Fonksiyona plot özelliği ekleyelim

def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#####################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# docstring
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, nümerik ve kategorik fakat kardinal(nicel) değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        Nümerik fakat kategorik olan değişkenler için sınıf eşik değeridir.
    car_th: int, float
        Kategorik fakat kardinal değişkenler için sınıf eşik değeridir.

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car:list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde
    """

    #cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f"cat_cols:{len(cat_cols)}")
    print(f"cat_cols:{len(num_cols)}")
    print(f"cat_but_cols:{len(cat_but_car)}")
    print(f"num_bat_cat:{len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# bonus
df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes=="bool":
        df[col]= df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#####################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#####################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()
df["survived"].value_counts()
cat_summary(df, "survived")

# Hedef değişkenin kategorik değişkenler ile analizi

df.groupby("sex").agg({"survived":"mean"})

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "class")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)
    print("\n")

# Hedef değişkenin sayısal değişkenler ile analizi

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)
    print("\n")

#####################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#####################################
# Korelasyon: Değişkenlerin birbiriyle ilişkisini ifade eden istatiksel bir ölçümdür.
# (-1, 1) aralığında yer alır. -1 veya 1'e yaklaştıkça ilşkinin şiddeti kuvvetlenir.
# 0'a yakın bir korelasyon değeri korelasyonun olmadığını gösterir.
# Eğer iki değişkenin arasındaki ilişki pozitifse buna pozitif korelasyon denir.
# Pozitif korelasyonda bir değişkenin değeri arttıkça diğerinin de değeri artar.
# Eğer iki değişkenin arasındaki ilişki negatifse buna negatif korelasyon denir.
# Negatif korelasyonda bir değişkenin değeri arttıkça diğerinin de değeri azalır.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()
df.info()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()

sns.set(rc = {"figure.figsize":(12,12)})
sns.heatmap(corr, cmap="RdBu") #korelasyonu gözlemleyebilmek için ısı haritası oluşturduk
plt.interactive(False)
plt.show()


# Yüksek Korelasyonlu Değişkenlerin Silinmesi

cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

cor_matrix[drop_list]
df.drop(drop_list, axis = 1)
df.shape

def high_correlated_cols(dataframe, plot=False, corr_th= 0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc = {"figure.figsize":(12,12)})
        sns.heatmap(corr, cmap="RdBu")
        plt.interactive(False)
        plt.show()
    return drop_list

high_correlated_cols(df)

drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis = 1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 400'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/competitions/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)

