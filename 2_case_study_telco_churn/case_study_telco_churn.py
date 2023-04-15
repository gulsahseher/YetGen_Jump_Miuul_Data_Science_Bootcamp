# #####################################################
# Telco Churn Feature Engineering
# #####################################################

#####################################
# İş Problemi
#####################################

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi
# ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

#####################################
# Veri Seti Hikayesi
#####################################

# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Feature_Engineering/2_case_study_telco_churn/datasets/Telco-Customer-Churn.csv")

df.info()
# df.value_counts()
df.isnull().sum()
df.isnull().values.any()
df.shape
# df.describe().T

df.head()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

#####################################
# Görev 1 : Keşifçi Veri Analizi
#####################################

# Adım 1: Genel resmi inceleyiniz.

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


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

# Kategorik değişken analizi

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df,col)

####################################################################

# Numerik değişken analizi

def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.interactive(False)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot="True")


# Adım 4 :Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

# Kategorik değişkenlere göre hedef değişkenin ortalaması

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean(),
                        "Count":dataframe.groupby(categorical_col)[target].count(),
                        "Ratio":100* dataframe[categorical_col].value_counts()/ dataframe.shape[0]}))

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)
    print("\n")

####################################################################

# Hedef değişkene göre numerik değişkenlerin ortalaması

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))


for col in num_cols:
    target_summary_with_num(df, "Churn", col)
    print("\n")

# Adım 5: Korelasyon analizi yapınız.

corr = df[num_cols].corr()

sns.set(rc = {"figure.figsize":(12,12)})
sns.heatmap(corr, fmt=".2f", annot=True, cmap="RdBu") # korelasyonu gözlemleyebilmek için ısı haritası oluşturduk
plt.interactive(False)
plt.show()

# TotalCharges ile tenure arasında yüksek korelasyon görülmektedir.

df.corrwith(df["Churn"]).sort_values(ascending=False)

# Adım 6: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1=dataframe[col_name].quantile(q1)
    quartile3=dataframe[col_name].quantile(q3)
    interquantile_range= quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df,col))

def grab_outliers(dataframe, col_name, index=False):
    low, up= outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name]<low)|(dataframe[col_name]>up))].shape[0]>10:
        print(dataframe[((dataframe[col_name]<low)|(dataframe[col_name]>up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index=dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(df, col, index=True)

# Adım 7: Eksik gözlem analizi yapınız.

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df, na_name = True)

#####################################
# Görev 2 : Feature Engineering
#####################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.isnull().sum()

# Adım 2: Encoding ve Base Model işlemlerini gerçekleştiriniz.

from catboost import CatBoostClassifier

dff = df.copy()

cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

dff.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols)

# dff.head()

y = dff["Churn"]
X = dff.drop(["customerID", "Churn"], axis=1)

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.3, random_state = 17)

catboost_model = CatBoostClassifier(verbose = False, random_state = 12345).fit(X_train, y_train)

y_pred = catboost_model.predict(X_test)

print("Accuracy:{:.2f}".format(accuracy_score(y_pred, y_test)))
print("Recall:{:.4f}".format(recall_score(y_pred, y_test)))
print("Precision:{:.4f}".format(precision_score(y_pred, y_test)))
print("F1:{:.4f}".format(f1_score(y_pred, y_test)))
print("AUC:{:.4f}".format(roc_auc_score(y_pred, y_test)))

# Accuracy:0.78
# Recall:0.6331
# Precision:0.4930
# F1:0.5544
# AUC:0.7292

# Adım 3: Yeni değişkenler oluşturunuz.

# Tenure değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"] >= 0 ) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24) , "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36) , "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60) , "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72) , "NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 yıl veya 2 yıl olan müşterileri Engaged olarak belirleme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply( lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply( lambda x: 1 if (x["NEW_Engaged"] == 0) or (x["SeniorCitizen"] == 0) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df["New_TotalService"] = (df[["PhoneService", "InternetService", "OnlineSecurity",
                              "OnlineBackup","DeviceProtection", "TechSupport",
                              "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

# Herhangi bi streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply( lambda x: 1 if (x["StreamingMovies"] == "Yes") or (x["StreamingTV"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df.apply( lambda x: 1 if (x["PaymentMethod"] == "Bank transfer (automatic)") or (x["PaymentMethod"] == "Credit card (automatic)") else 0, axis=1)

# Ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] +1) # tenureun 0 olduğu değerler için +1 eklendi

# Güncel fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df["New_TotalService"] +1)

df.head()
df.shape

# Adım 4: Label Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# Adım 5: One Hot Encoding

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# Adım 6: Model oluşturunuz.

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print("Accuracy:{:.2f}".format(accuracy_score(y_pred, y_test)))
print("Recall:{:.4f}".format(recall_score(y_pred, y_test)))
print("Precision:{:.4f}".format(precision_score(y_pred, y_test)))
print("F1:{:.4f}".format(f1_score(y_pred, y_test)))
print("AUC:{:.4f}".format(roc_auc_score(y_pred, y_test)))

# Accuracy:0.79
# Recall:0.6535
# Precision:0.4895
# F1:0.5598
# AUC:0.7397

df.head()

# Değişkenlerin model başarımını arttırmada ne kadar güçlü olduğunu gösteren grafiğin çıkarılması

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    f1_df = pd.DataFrame(data)

    f1_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    plt.figure(figsize=(15, 10))
    sns.barplot(x = f1_df["feature_importance"], y=f1_df["feature_names"])

    plt.title(model_type + " FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, "CATBOOST")



