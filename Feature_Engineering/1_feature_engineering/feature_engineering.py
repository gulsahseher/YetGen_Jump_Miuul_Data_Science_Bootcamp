##################################################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
##################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

def load_application_train():
    data = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Feature_Engineering/1_feature_engineering/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Feature_Engineering/1_feature_engineering/datasets/titanic.csv")
    return data

df = load()
df.head()

##################################################################
# 1. Outliers (Aykırı Değerler)
##################################################################

sns.boxplot(x=df["Age"])
plt.show()

# 1. Eşik değer belirle.
# 2. Aykırı değerlere eriş.
# 3. Hızlıca aykırı değer var mı yok mu diye ara.

#####################################
# Aykırı Değerleri Yakalama
#####################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

iqr = q3 - q1
up = q3 + iqr * 1.5
low = q1 - iqr * 1.5 # negatif değerler yer almadığı için negatif bir değer geldi

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index

#####################################
# Aykırı Değer Var Mı Yok Mu?
#####################################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)
df[(df["Age"] < low)].any(axis=None)

#####################################
# Fonksiyonlaştırma
#####################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1=dataframe[col_name].quantile(q1)
    quartile3=dataframe[col_name].quantile(q3)
    interquantile_range= quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")
df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low) | (df["Fare"] > up)].index

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Fare")
check_outlier(df, "Age")

# Verisetinde gezerek sayısal olmayan değişkenleri yakalayan fonksiyonu yazalım: grab_col_names

dff = load_application_train()
dff.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ----------
    dataframe: dataframe
            Değişken isimleri alınmak istenilen dataframe
    cat_th: int, optional
            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, optional
            kategorik fakat kardinal değişkenler için sınıf eşik değeri

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

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

grab_col_names(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]
num_cols

for col in num_cols:
    print(col, check_outlier(df, col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff, col))

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
num_cols

for col in num_cols:
    print(col, check_outlier(dff, col))

#####################################
# Aykırı Değerlerin Kendilerine Erişmek
#####################################

def grab_outliers(dataframe, col_name, index=False):
    low, up= outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name]<low)|(dataframe[col_name]>up))].shape[0]>10:
        print(dataframe[((dataframe[col_name]<low)|(dataframe[col_name]>up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index=dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age", index=True)
age_index = grab_outliers(df, "Age", index=True)

grab_outliers(df, "Fare", index=True)

#####################################
# Aykırı Değer Problemini Çözme
#####################################

# Silme

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

# Baskılama Yöntemi (re-assigment with thresholds)

low, up = outlier_thresholds(df, "Fare")

# df'teki "Fare" değişkenine ait olan aykırı değerler

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
df.loc[((df["Fare"] < low) | (df["Fare"] > up)) , "Fare"]

# Aykırı değerler yerine alt ve üst limitleri atayalım
df.loc[(df["Fare"] > up), "Fare"] = up
df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_threshold(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#####################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#####################################

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include = ["float64", "int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5] # okunabilirlik açısından negatif skorlar tercih edilir
# df_scores = -df_scores
np.sort(df_scores)[0:5]

# En dik eğimin olduğu nokta (en marjinal değişim) eşik değer (threshold) olarak belirlenebilir.
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

th = np.sort(df_scores)[3]
df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

##################################################################
# 2. Missing Values (Eksik Değerler)
##################################################################

#####################################
# Eksik Değerlerin Yakalanması
#####################################

df = load()
df.head()

# eksi gözlem var mı yok mu sorgusu
df.isnull().values.any()

# değişkenlerdeki eksik değer sayısı
df.isnull().sum()

# değişkenlerdeki tam değer sayısı
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

# en az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# azalan şekilde eksik değerleri sıralamak
df.isnull().sum().sort_values(ascending=False)

# eksik değerlerin gözlem birimlerine göre tüm veri setine oranları
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# eksik değere sahip değişkenlerin isimlerini yakalamak için
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, na_name=True)

#####################################
# Eksik Değer Problemini Çözme
#####################################

missing_values_table(df)

# Çözüm 1: Hızlıca silmek

df.dropna().shape

# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x:x.fillna(x.mean()), axis=0)
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum() # Kategorik değişkeni mode ile doldurduk

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# Kategorik Değişken Kırılımında Değer Atama

df.groupby("Sex")["Age"].mean()
df["Age"].mean()

# Eksik  yaş değerlerini cinsiyet kırılımında dolduralım. Yani yaş ortalamalarını cinsiyete göre alıp eksik değerleri cinsiyetlere göre doldurduk.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]
df.isnull().sum()

# Çözüm 3: Tahmine Dayalı Atama ile Doldurma

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) #drop_first kullanmak modelin verimi için önemlidir

dff.head()

# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]

#####################################
# Gelişmiş Analizler
#####################################

# Eksik veri yapısının incelenmesi

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

# Korelasyonun 1'e yakın olması pozitif yönlü bir kuvvetli ilişkiyi ifade eder. Korelasyonun -1'e yakın olması da negatif yönlü bir kuvvetli ilişkiyi ifade eder.
# Pozitif yönlü korelasyon değişkenlerdeki eksiklerin birlikte olduğu düşünülür. Yani birisinde eksiklik varken diğerinde de eksiklik vardır diyebiliriz.
# Negatif yönlü korelasyon varsa eksiklerin birbirinin zıttı olduğu düşünülür. Yani birisinde eksiklik varken diğerinde eksiklik yoktur denilebilir.
msno.heatmap(df)
plt.show()

#####################################
# Eksik Değerlerin Bağımlı Değişken İle İlişkisinin İncelenmesi
#####################################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)

##################################################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
##################################################################

#####################################
# Label Encoding & Binary Encoding
#####################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder() # Alfabetik sıraya göre ilk gördüğü değere 0 değerini verir.
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1]) # 0 ve 1 değişkenine karşılık gelen değerleri verir.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df, "Sex").head()

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2] # nunique() eksik değeri bir sınıf olarak görmez.

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]

df[binary_cols].head()

# Eksik değerler de etiketlenmiştir.
for col in binary_cols:
    label_encoder(df, col)

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique()) # Eşsiz değerler gözlemlenirken NaN da eşsiz bir değermiş gibi alınır.

#####################################
# One-Hot Encoding
#####################################

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head() # Alfabetik sıraya göre gelen ilk sınıf değişken olarak atanmaz.
                                                                 # Bu yüzden Embarked değişkeninin C sınıfı atanmamıştır.

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head() # Eksik değerler de sınıf olarak gelir.

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

df.head()

#####################################
# Rare Encoding
#####################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategorik ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts() # Academic degree sınıfı diğer sınıflara göre çok az

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
        plt.interactive(False)

for col in cat_cols:
    cat_summary(df, col, plot=True)

# 2. Rare kategorik ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

# 3. Rare encoder yazacağız.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

##################################################################
# 4. Feature Scaling (Özellik Ölçeklendirme)
##################################################################

#####################################
# StandardScaler (Z standartlaştırılması)
#####################################

# Klasik standartlaştırma. Ortalamayı çıkar standart sapmayı böl. z = (x - u) / s
# Standart sapma ve ortalama verisetindeki aykırı değerlerden etkilenir.

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#####################################
# RobustScaler
#####################################

# Medyanı çıkar, iqr'a böl.
# Aykırı değerlerden etkilenmez.

# df = load()
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#####################################
# MinMaxScaler
#####################################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0)) - X.min(axis=0)
# X_scaled = X_std * (max - min) + min
# 0 - 1 arasında değerler dağılır

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T
df.head()

age_cols = [col for col in df.columns if "Age" in col]

# Yapılarını koruyacak şekilde ifade ediliş tarzlarını düzenledik. Kullanılan 3 ölçeklendirmede de dağılımlar değişmemiştir.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        # plt.interactive(False)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

#####################################
# Numeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning
#####################################

# qcut() metodu bir değişkenin değerlerini küçükten büyüğe sıralar ve çeyrek değerlere göre 5 parçaya böler.

df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()

##################################################################
# 5. Feature Extraction (Özellik Çıkarımı)
##################################################################

#####################################
# Binary Features: Flag, Bool, True-False
#####################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")

df.groupby("NEW_CABIN_BOOL").agg({"Survived":"mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) #p -value değeri 0.05'ten küçük olduğu için iki değişken arasında anlamlı bir farklılık olduğu sonucuna ulaşılabilir.

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived":"mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

#####################################
# Text'ler Üzerinden Özellikler Türetmek
#####################################

df.head()
[row for row in df["Name"] if row in "Mr."]

# Letter Count

df["NEW_NAME_COUNT"] = [len(row) for row in df["Name"]]
# df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

# Word Count

df["NEW_NAME_WORD_COUNT"] = [len((row).split(" ")) for row in df["Name"]]
# df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

# Özel Yapıları Yakalamak

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

# Regex ile Değişken Türetmek

df.head(30)

df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#####################################
# Date Değişkenleri Üretmek
#####################################

dff = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Feature_Engineering/1_feature_engineering/datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff["year"] = dff["Timestamp"].dt.year

# month
dff["month"] = dff["Timestamp"].dt.month

# year diff
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff["month_diff"] =  (date.today().year - dff["Timestamp"].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

# day name
dff["day_name"] = dff["Timestamp"].dt.day_name()

dff.head()

#####################################
# Feature Interactions (Özellik Etkileşimleri)
#####################################

# Değişkenlerin birbiri ile etkileşime girmesi demektir. Örneğin 2 değişkenin çarpılması, toplanması gibi.

df.load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] * df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 21) & (df["Age"] <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50) , "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 21) & (df["Age"] <= 50), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50) , "NEW_SEX_CAT"] = "seniorfemale"

df.groupby("NEW_SEX_CAT")["Survived"].mean()

##################################################################
# Titanic Uçtan Uca Uygulama: Feature Engineering & Data Preprocessing (End-to-End Application)
##################################################################

df = load()
df.head()
df.shape
df.info()

df.columns = [col.upper() for col in df.columns]

#####################################
# 1. Feature Engineering (Değişken Mühendisliği)
#####################################

# Cabin Bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype("int")

# Name Count
df["NEW_NAME_COUNT"] = [len(row) for row in df["NAME"]]

# name word count
df["NEW_NAME_WORD_COUNT"] = [len((row).split(" ")) for row in df["NAME"]]

# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# name title
df["NEW_TITLE"] = df.NAME.str.extract(" ([A-Za-z]+)\.", expand=False)

# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] * df["PARCH"] + 1

# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# is alone
df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"

# age level
df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

# sex x age
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50) , "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50) , "NEW_SEX_CAT"] = "seniorfemale"

df.head()
df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#####################################
# 2. Outliers (Aykırı Değerler)
#####################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_threshold(df, col)

#####################################
# 3. Missing Values (Eksik Değerler)
#####################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50) , "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50) , "NEW_SEX_CAT"] = "seniorfemale"

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#####################################
# 4. Label Encoding
#####################################

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#####################################
# 5. Rare Analyser
#####################################

rare_analyser(df, "SURVIVED", cat_cols)
df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#####################################
# 6. One-Hot Encoding
#####################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in 'PASSENGERID']
num_cols

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis = None)]

# df.drop(useless_cols, axis=1, inplace=True)

#####################################
# 7. Standart Scaler
#####################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

#####################################
# 8. Model
#####################################
from sklearn.ensemble import RandomForestClassifier

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) # 0.8059701492537313

#####################################
# Hiç bir işlem yapılmadan elde edilecek skor
#####################################

dff = load()
dff.dropna(inplace=True)

dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)

y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_pred, y_test) # 0.7090909090909091

# Yeni türettiğimiz değişkenlerin modeldeki önemi

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value":model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importance.png")

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=17)

rf_tuned = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_pred, y_test)

plot_importance(rf_tuned, X_train)