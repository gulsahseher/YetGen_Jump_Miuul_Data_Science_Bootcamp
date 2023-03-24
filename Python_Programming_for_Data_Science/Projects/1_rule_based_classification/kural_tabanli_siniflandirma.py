##################################################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
##################################################################

#####################################
# İş Problemi
#####################################

# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak
# seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler
# oluşturup bu segmentlere göre yeni gelebilecek müşterilerin
# şirkete ortalama ne kadar kazandırabileceğini tahmin etmek
# istemektedir.

#####################################
# Değişkenler
#####################################

# PRICE: Müşterinin harcama tutarı
# SOURCE: Müşterinin bağlandığı cihaz türü
# SEX: Müşterinin cinsiyeti
# COUNTRY: Müşterinin ülkesi
# AGE: Müşterinin yaşı

#####################################
# Proje Görevleri
#####################################

# Hedeflenen çıktı

#  customers_level_based     PRICE        SEGMENT
# BRA_ANDROID_FEMALE_0_18    35.6453         B
# BRA_ANDROID_FEMALE_19_23   34.0773         C
# BRA_ANDROID_FEMALE_24_30   33.8639         C
# BRA_ANDROID_FEMALE_31_40   34.8983         B
# BRA_ANDROID_FEMALE_41_66   36.7371         A

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#####################################
# Görev 1: Aşağıdaki Soruları Yanıtlayınız
#####################################

#  Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

df = pd.read_csv("dataset/persona.csv")
df.info()
df.head()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?

df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df.columns
df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY")["PRICE"].sum()

# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()

#####################################
# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#####################################

# Elde edilmesi gereken çıktı:

# COUNTRY  SOURCE   SEX     AGE
# bra      android  female  15     38.714286
#                           16     35.944444
#                           17     35.666667
#                           18     32.255814
#                           19     35.206897
#                                    ...
# usa      ios      male    42     30.250000

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()

#####################################
# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
#####################################

# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

# Elde edilmesi gereken çıktı:

# COUNTRY  SOURCE   SEX     AGE
# bra      android  male    46     59.0
# usa      android  male    36     59.0
# fra      android  female  24     59.0
# usa      ios      male    32     54.0
# deu      android  female  36     49.0

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False)

#####################################
# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#####################################

# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.

agg_df.reset_index().head()
agg_df = agg_df.reset_index()
agg_df.head()

#####################################
# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
#####################################

# Elde edilmesi gereken çıktı

# COUNTRY   SOURCE   SEX     AGE  PRICE AGE_CAT
# bra       android  male    46   59.0  41_70
# usa       android  male    36   59.0  31_40
# fra       android  female  24   59.0  24_30
# usa       ios      male    32   54.0  31_40
# deu       android  female  36   49.0  31_40

# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'

agg_df["AGE"].describe()

bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

my_labels = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=my_labels)

agg_df.head()

#####################################
# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
#####################################

# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

# Elde edilmesi gereken çıktı

# customers_level_based     PRICE
# BRA_ANDROID_MALE_41_66    59.0
# USA_ANDROID_MALE_31_40    59.0
# FRA_ANDROID_FEMALE_24_30  59.0
# USA_IOS_MALE_31_40        54.0
# DEU_ANDROID_FEMALE_31_40  49.0

col_names = ["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]

[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[3].upper() for row in agg_df[col_names].values]

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[3].upper() for row in agg_df[col_names].values]
agg_df.head()

# agg_df'te sadece "customers_level_based", "PRICE" değişkenlerinin yer almasını sağlayalım.
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()

# Her bir "customers_level_based" değişkeninden 1 tane yer alması için "PRICE" ortalamalarını alalım.
agg_df["customers_level_based"].value_counts()
agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE":"mean"})
agg_df = agg_df.reset_index()
agg_df

# Yapılan işlemi doğrulamak için value_counts ile kontrol edelim
agg_df["customers_level_based"].value_counts()

#####################################
# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız
#####################################

# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D", "C", "B", "A"])
agg_df.head()

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

#####################################
# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz
#####################################

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]