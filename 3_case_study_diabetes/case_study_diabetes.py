# #####################################################
# Feature Engineering
# #####################################################

#####################################
# İş Problemi
#####################################

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.

#####################################
# Veri Seti Hikayesi
#####################################

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Pregnancies: Hamilelik sayısı
# Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness: Cilt Kalınlığı
# Insulin: 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI: Vücut kitle endeksi
# Age: Yaş (yıl)
# Outcome: Hastalığa sahip (1) ya da değil (0)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Feature_Engineering/3_case_study_diabetes/datasets/diabetes.csv")

df.info()
# df.value_counts()
df.isnull().sum()
df.isnull().values.any()
df.shape
# df.describe().T

df.head()

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
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.interactive(False)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot="True")

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

# Kategorik değişkenlere göre hedef değişkenin ortalaması

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean(),
                        "Count":dataframe.groupby(categorical_col)[target].count(),
                        "Ratio":100* dataframe[categorical_col].value_counts()/ dataframe.shape[0]}))

for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)
    print("\n")

####################################################################

# Hedef değişkene göre numerik değişkenlerin ortalaması

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)
    print("\n")

# Adım 5: Korelasyon analizi yapınız.

corr = df[num_cols].corr()

sns.set(rc = {"figure.figsize":(18,12)})
sns.heatmap(corr, fmt=".2f", annot=True, cmap="RdBu") # korelasyonu gözlemleyebilmek için ısı haritası oluşturduk
plt.yticks(rotation=0)
plt.interactive(False)
plt.show()

# TotalCharges ile tenure arasında yüksek korelasyon görülmektedir.

df.corrwith(df["Outcome"]).sort_values(ascending=False)

# Adım 6: Base Model Kurulumu

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

print("Accuracy:{:.4f}".format(accuracy_score(y_pred, y_test)))
print("Recall:{:.4f}".format(recall_score(y_pred, y_test)))
print("Precision:{:.4f}".format(precision_score(y_pred, y_test)))
print("F1:{:.4f}".format(f1_score(y_pred, y_test)))
print("AUC:{:.4f}".format(roc_auc_score(y_pred, y_test)))

# Accuracy:0.7706
# Recall:0.7059
# Precision:0.5926
# F1:0.6443
# AUC:0.7517

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature":features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, X)


#####################################
# Görev 2 : Feature Engineering
#####################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.

# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# Gözlem birimlerinde 0 olan değişkenlerin her birisine gidip 0 içeren gözlem değerlerini NaN ile değiştirdik.

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col]) # df[col] == 0 ise NaN ile doldur içermiyorsa df[col] olarak kalsın.

# Adım 2: Eksik gözlem analizi yapınız.

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_cols = missing_values_table(df, na_name = True)

# Eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_cols)

# Eksik değerlerin doldurulması

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

# Adım 3: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

def replace_with_threshold(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_threshold(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

# Adım 4: Özellik Çıkarımı

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturma
df.loc[((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI 18.5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası overweight ve 30 üstü obese
df["NEW_BMI"] = pd.cut(x = df["BMI"], bins = [0, 18.5, 24.9, 29.9, 100],
                       labels = ["Underweight", "Normal", "Overweight", "Obese"])

# Glukoz değerini kategorik değişkenlere çevirme
df["NEW_GLUCOSE"] = pd.cut(x = df["Glucose"], bins = [0, 140, 200, 300],
                           labels = ["Normal", "Prediabetes", "Diabetes"])

# Yaş ve beden kitle indeksini birarada düşünerek kategorik değişken oluşturma
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >=50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] >= 30) & ((df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] >= 30) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

#  Yaş ile glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Glucose"] < 70) & ((df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 140) & (df["Age"] >=50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >=50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >=50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >=50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

# İnsülin değeri ile kategorik değişken türetme
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

df.columns = [col.upper() for col in df.columns]

df.head()
df.shape

# Adım 5: Encoding işlemlerini gerçekleştiriniz.

# Label Encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# One-Hot Encoding İşlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# Standartlaştırma

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

# Adım 6: Modelleme yapınız.

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

print("Accuracy:{:.4f}".format(accuracy_score(y_pred, y_test)))
print("Recall:{:.4f}".format(recall_score(y_pred, y_test)))
print("Precision:{:.4f}".format(precision_score(y_pred, y_test)))
print("F1:{:.4f}".format(f1_score(y_pred, y_test)))
print("AUC:{:.4f}".format(roc_auc_score(y_pred, y_test)))

# Accuracy:0.7792
# Recall:0.7027
# Precision:0.6420
# F1:0.6710
# AUC:0.7590

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature":features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, X)

