##################################################################
# Fonksiyonlara Özellik ve Docstring Ekleme
##################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

#####################################
# Fonksiyonlara Özellik Eklemek
#####################################

# Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argümanla biçimlendirilebilir olsun. Var olan
# özelliği de argümanla kontrol edilebilir hale getirebilirsiniz.

def cat_summary(dataframe, col_name, plot = False, value_counts=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

    if value_counts:
        print(dataframe.value_counts())

cat_summary(df, "sex", value_counts=True)


#####################################
# Docstring Yazımı
#####################################

# Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi (uygunsa) barındıran numpy tarzı docstring
# yazınız. (task, params, return, example)

def check_df(dataframe, head=5):
    """
    Dataframe hakkında shape, type, boş elemanlarının toplamı gibi genel özelliklerini sıralamaya yarayan fonksiyondur.

    Parameters
    ----------
    dataframe: dataframe
    head: dataframe'de yer alan ilk 5 satır

    Returns
    -------
    None

    Examples:
    ------
    check_df(df, 7)
    """

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

check_df(df, head=5)

def cat_summary(dataframe, col_name, plot = False):
    """
    Her bir değişkene ait olan değişken sayılarını ve bu değişkenlerin kategorilerine
    göre olan ortalamalarını içerir. plot=True olduğu takdirde değişkenlere ait grafikler de
    çizdirilebilir.

    Parameters
    ----------
    dataframe: dataframe
    col_name: sütun ismi
    plot: grafik çizdirme
    Ön tanımlı değeri False'tur. True olduğunda değişkene ait grafik çizdirilir.

    Returns
    -------
    None

    Examples:
    ------
    import seaborn as sns
    df = sns.load_dataset("titanic")
    cat_summary(df, "Survived", plot=True)
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#############################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex")

import random

def toss_coin():
    T = random.randrange(0, 2)
    H = random.randrange(0, 2)
    return T, H

toss_coin()

from functools import reduce
num_list = np.arange(10)
filter_list=list(filter(lambda x: x%3 ==0, num_list))
final_list=reduce(lambda x, y:x*y, filter_list)

df[["sex", "survived"]].groupby("sex")

import seaborn as sns
titanic=sns.load_dataset("titanic")
sns.countplot(x="class", data=titanic)
plt.show()

titanic.loc[:,titanic.columns.str.contains("a")]