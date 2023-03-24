# #####################################################
# List Comprehension ve Pandas Alıştırmalar
# #####################################################

################
# Görev 1
################

# car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz
# ve başına NUM ekleyiniz.

# Beklenen çıktı:
# ['NUM_TOTAL',
#  'NUM_SPEEDING',
#  'NUM_ALCOHOL',
#  'NUM_NOT_DISTRACTED',
#  'NUM_NO_PREVIOUS',
#  'NUM_INS_PREMIUM',
#  'NUM_INS_LOSSES',
#  'ABBREV']

# Not
# Numeric olmayan değişkenlerin de isimleri büyümeli.
# Tek bir list comprehension yapısı kullanılmalı

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

# num_cols = [col for col in df.columns if df[col].dtype != "O"]

["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

################
# Görev 2
################

# List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
# değişkenlerin isimlerinin sonuna "FLAG" yazınız.

# Beklenen çıktı:
# ['TOTAL_FLAG',
#  'SPEEDING_FLAG',
#  'ALCOHOL_FLAG',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM_FLAG',
#  'INS_LOSSES_FLAG',
#  'ABBREV_FLAG']

# Not
# Tüm değişkenlerin isimleri büyük harf olmalı.
# Tek bir list comprehension yapısı ile yapılmalı

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns ]

################
# Görev 3
################

# List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
# değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

og_list = ["abbrev", "no_previous"]

# Beklenen çıktı
#    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# 0   18.8     7.332    5.640          18.048       784.55      145.08
# 1   18.1     7.421    4.525          16.290      1053.48      133.93
# 2   18.6     6.510    5.208          15.624       899.47      110.35
# 3   22.4     4.032    5.824          21.056       827.34      142.39
# 4   12.0     4.200    3.360          10.920       878.41      165.63

# Not
# Önce verilen listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz ve adını new_df olarak isimlendiriniz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

new_cols = [col for col in df.columns if col not in og_list]

new_df=df[new_cols]
new_df.head()

################
# Pandas Alıştırmalar
################

import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

#seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset("titanic")
df.head()
df.shape

#titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz
df["sex"].value_counts()

#Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.nunique()

# pclass değişkeninin unique değerlerinin sayısını bulunuz.
df["pclass"].unique()

#pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
df[["pclass","parch"]].nunique()

# embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()

# embarked değeri C olanların tüm bilgelerini gösteriniz.
df.loc[(df["embarked"] == "C"), :]
df[df["embarked"] == "C"]

# embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df.loc[(df["embarked"] != "S"), :]
df[df["embarked"] != "S"]["embarked"].unique()

# Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df.loc[(df["age"] < 30) & (df["sex"] == "female"), :]
df[(df["age"] < 30) & (df["sex"] == "female")]

# Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df.loc[(df["fare"] > 500) | (df["age"] > 70), :]
df[(df["fare"] > 500) | (df["age"] > 70)]

# Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()

# who değişkenini dataframe’den çıkarınız.
df.drop("who", axis=1).head()

# deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()

# age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"].median()
df["age"].fillna(df["age"].median(), inplace=True)
df["age"].isnull().sum()

# survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby(["pclass","sex"]).agg({ "survived":["sum","count","mean"]})

# 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
# setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

def age_30(age):
    if age < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))
df.head()

# Hiç fonksiyon yazmadan uygulamak için:
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
import seaborn as sns
import pandas as pd
df = sns.load_dataset("tips")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby("time").agg({'total_bill':["min", "max", "mean"]})

# Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby(["day","time"]).agg({"total_bill":["sum","min", "max", "mean"]})

# Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby(["day"]).agg({'total_bill':["sum", "min", "max", "mean"],
                                                                            "tip":["sum","min", "max", "mean"]})

#  size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
df.loc[(df["size"] < 3) & (df["total_bill"] > 10),"total_bill"].mean()

# total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape
