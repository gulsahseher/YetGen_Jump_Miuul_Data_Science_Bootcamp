##################################################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
##################################################################

#####################################
# İş Problemi
#####################################

# Gezinomi yaptığı satışların bazı özelliklerini kullanarak seviye tabanlı
# (level based) yeni satış tanımları oluşturmak ve bu yeni satış
# tanımlarına göre segmentler oluşturup bu segmentlere göre yeni
# gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceğini
# tahmin etmek istemektedir.

#####################################
# Değişkenler
#####################################

# SaleId : Satış id
# SaleDate : Satış Tarihi
# Price : Satış için ödenen fiyat
# ConceptName: Otel konsept bilgisi
# SaleCityName: Otelin bulunduğu şehir bilgisi
# CheckInDate : Müşterinin otelegirişitarihi
# CInDay:Müşterinin otele giriş günü
# SaleCheckInDayDiff: Check in ile giriş tarihi gün farkı
# Season: Otele giriş tarihindeki sezon bilgisi

#####################################
# Proje Görevleri
#####################################

# Hedeflenen çıktı

# sales_level_based          SEGMENT    Price
# GIRNE_HERŞEY DAHIL_HIGH       A      103.9354
# GIRNE_HERŞEY DAHIL_LOW        A      90.93594
# İZMIR_YARIM PANSIYON_HIGH     A      87.6573
# DIĞER_HERŞEY DAHIL_LOW        A      87.31088
# DIĞER_HERŞEY DAHIL_HIGH       A      83.78727


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#####################################
# Görev 1: Aşağıdaki Soruları Yanıtlayınız
#####################################

# Soru 1 : miuul_gezinomi.xlsx dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

df = pd.read_excel("dataset/miuul_gezinomi.xlsx")
df.head()
df.info()
df.shape
# Soru 2: Kaç unique şehir vardır? Frekansları nedir?

df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()

# Soru 3: Kaç unique Concept vardır?

df["ConceptName"].nunique()

# Soru 4: Hangi Concept’den kaçar tane satış gerçekleşmiş?

df["ConceptName"].value_counts()

# Soru 5: Şehirlere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("SaleCityName")["Price"].sum()
df.groupby("SaleCityName").agg({"Price": "sum"})

# Soru 6: Concept türlerine göre göre ne kadar kazanılmış?

df.groupby("ConceptName")["Price"].sum()
df.groupby("ConceptName").agg({"Price": "sum"})

# Soru 7: Şehirlere göre PRICE ortalamaları nedir?

df.groupby("SaleCityName")["Price"].mean()
df.groupby("SaleCityName").agg({"Price": "mean"})

# Soru 8: Conceptlere göre PRICE ortalamaları nedir?

df.groupby("ConceptName")["Price"].mean()
df.groupby("ConceptName").agg({"Price": "mean"})

# Soru 9: Şehir-Concept kırılımında PRICE ortalamaları nedir?

df.groupby(["SaleCityName","ConceptName"])["Price"].mean()
df.groupby(["SaleCityName","ConceptName"]).agg({"Price": "mean"})

#####################################
# Görev 2: SaleCheckInDayDiff değişkenini kategorik bir değişkene çeviriniz.
#####################################

# SaleCheckInDayDiff değişkeni müşterinin CheckIn tarihinden ne kadar önce satin alımını tamamladığını gösterir.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_7’, ‘7_30', ‘30_90', ‘90_max’ aralıklarını kullanabilirsiniz.
# Bu aralıklar için "Last Minuters", "Potential Planners", "Planners", "Early Bookers“ isimlerini kullanabilirsiniz

bins= [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()]

# my_labels = ["0_7", "7_30", "30_90", "90_" + str(df["SaleCheckInDayDiff"].max())]

labels  = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

# pd.cut(df["SaleCheckInDayDiff"], bins, labels=my_labels)

df["EB_Score"]= pd.cut(df["SaleCheckInDayDiff"], bins, labels=labels)
df.head()

#####################################
# Görev 3: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#####################################

# Şehir-Concept-EB Score, Şehir-Concept- Sezon, Şehir-Concept-CInDay kırılımında ortalama ödenen ücret ve yapılan işlem sayısı cinsinden
# inceleyiniz.

# Elde edilmesi gereken çıktı:

#                                                     PRICE
#    SaleCityName    ConceptName       EB_Score    mean  count
# 0       Antalya    Herşey Dahil   Last Minuters  62.75 14148
# 1                                     Potential
#                                        Planners  64.90 8874
# 2                                      Planners  67.88 4490
# 3                                 Early Bookers  66.49 3281
# 4                  Oda + Kahvaltı Last Minuters  65.35 503

df.groupby(["SaleCityName", "ConceptName","EB_Score"]).agg({"Price":["mean","count"]}).head()

df.groupby(["SaleCityName", "ConceptName","Seasons"]).agg({"Price":["mean","count"]}).head()

df.groupby(["SaleCityName", "ConceptName","CInDay"]).agg({"Price":["mean","count"]}).head()

#####################################
# Görev 4: City-Concept-Season kırılımının çıktısını PRICE'a göre sıralayınız
#####################################

# Elde edilmesi gereken çıktı:

#                                               PRICE
#           SaleCityName  ConceptName   Seasons
# 0            Girne      Herşey Dahil     High 103.94
# 1                                         Low 90.94
# 2            İzmir      Yarım Pansiyon   High 87.66
# 3            Diğer      Herşey Dahil      Low 87.31
# 4                                        High 83.79

# Elde ettiğiniz çıktıyı agg_df olarak kaydediniz

agg_df = df.groupby(["SaleCityName","ConceptName","Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)

#####################################
# Görev 5: Indekste yer alan isimleri değişken ismine çeviriniz.
#####################################

agg_df = agg_df.reset_index()
agg_df.head()

#####################################
# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
#####################################

# Yeni seviye tabanlı satışları tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: sales_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek sales_level_based değişkenini oluşturmanız gerekmektedir.

# Elde edilmesi gereken çıktı:
#
# SaleCityName ConceptName     Seasons    PRICE     sales_level_based
# Girne        Herşey Dahil     High     103.9354  GIRNE_HERŞEY DAHIL_HIGH
# Girne        Herşey Dahil      Low     90.93594  GIRNE_HERŞEY DAHIL_LOW
# İzmir        Yarım Pansiyon   High     87.6573   İZMIR_YARIM PANSIYON_HIGH
# Diğer        Herşey Dahil      Low     87.31088  DIĞER_HERŞEY DAHIL_LOW
# Diğer        Herşey Dahil     High     83.78727  DIĞER_HERŞEY DAHIL_HIGH

agg_df["sales_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() for row in agg_df.values]
agg_df.head()

#####################################
# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
#####################################

# Yeni personaları PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels = ["D", "C", "B", "A"])
agg_df.head()
agg_df.groupby("SEGMENT").agg({"Price":["mean", "max", "sum"]})

#####################################
# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
#####################################

# Antalya’da herşey dahil ve yüksek sezonda tatil yapmak isteyen bir kişinin ortalama ne kadar gelir kazandırması beklenir?
agg_df.sort_values(by="Price")

new_user="ANTALYA_HERŞEY DAHIL_HIGH"
agg_df[agg_df["sales_level_based"]==new_user]

# Girne’de yarım pansiyon bir otele düşük sezonda giden bir tatilci hangi segmentte yer alacaktır?

agg_df.sort_values(by="Price")

new_user="GIRNE_YARIM PANSIYON_LOW"
agg_df[agg_df["sales_level_based"]==new_user]