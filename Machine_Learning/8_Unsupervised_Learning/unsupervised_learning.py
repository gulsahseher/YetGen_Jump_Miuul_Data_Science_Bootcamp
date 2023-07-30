# #####################################################
# Unsupervised Learning
# #####################################################

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV

#####################################
# K-Means
#####################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/USArrests.csv", index_col=0)
df.head()
df.isnull().sum()
df.info
df.describe().T

# Uzaklık temelli ve Gradient Descent temelli yöntemlerin kullanımındaki süreçlerde değişkenlerin standartlaştırılması gerekmektedir.

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5] # numpy array haline geldiği için ilk 5 değere head ile bakılamaz

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters # küme sayısı
kmeans.cluster_centers_ # kümenin merkezleri
kmeans.labels_ # verisetindeki her bir gözlemin k-means tarafından belirlenen etiketleri
kmeans.inertia_ # Örneklerin en yakın küme merkezine uzaklıklarının karelerinin toplamı (SSE, SSR).

# Optimum Küme Sayısını Belirleme

# Değişken sayısı kadar küme oluşturulmamalı. Sadece SSE'ye bakılarak küme sayısına karar verilemez.
# İş bilgisinin de küme sayısını belirlerken göz önünde bulundurulması gerekir.

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları (SSE/SSR/SSD)")
plt.title("Optimum Küme Sayısı İçin Elbow Yöntemi")
plt.show()

# Elbow yöntemine göre karar verilirken dirseklenmenin yani eğimin en fazla olduğu noktalar seçilir.

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_ # optimum küme sayısı

# Final Cluuster'ların Oluşturulması

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/USArrests.csv", index_col=0)

df["Clusters"] = clusters_kmeans
df["Clusters"] = df["Clusters"] + 1
df.head()

df[df["Clusters"]==5]

df.groupby("Clusters").agg(["count", "mean", "median"])

df.to_csv("clusters.csv") # kümeleme sonuçlarını kaydetme

#####################################
# Hierarchical Clustering
#####################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 20))
df = sc.fit_transform(df)

hc_average = linkage(df, "average") # Birleştirici bir şekilde kümeleme işlemini gerçekleştirir.

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode= "lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.savefig("h_clustering.png")
plt.show()

# Optimum Küme Sayısını Belirleme

plt.figure(figsize=(7, 5))
plt.title("Dendograms")
dend = dendrogram(hc_average)
plt.axhline(y=12.5, color = "r", linestyle = "--")
plt.axhline(y=10, color = "b", linestyle = "--")
# plt.savefig("h_clustering.png")
plt.show()

# Final Modeli Oluşturma

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

cluster = cluster.fit_predict(df)

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/USArrests.csv", index_col=0)
df["Hi_Cluster_NO"] = cluster
df["Hi_Cluster_NO"] = df["Hi_Cluster_NO"] + 1

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1

#####################################
# Principal Component Analysis
#####################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/Hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df) # Bileşenlerin başarısı bileşenlerin açıkladığı varyans oranlarına göre belirlenmektedir.

pca.explained_variance_ratio_ # PCA'in oluşturduğu 16 adet yeni bileşenin hepsinin açıkladığı varyans oranı
np.cumsum(pca.explained_variance_ratio_) # Bileşenlerin bir araya gelmesi sonucunda toplam ne kadar açıklama oranına sahip
                                         # olduğuna bakılır.

# Optimum Bileşen Sayısı

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_)) # 3 bileşen tercih edilebilir
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

# Final PCA'in Oluşturulması

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_) # Son durumda verisetindeki bileşenlerin %82'si açıklanabilmiştir.

#####################################
# BONUS: Principal Component Regression
#####################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/Hitters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]), df[others]], axis=1)
final_df.head()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error"))) # 345.6021106351967
y.mean() # 535.9258821292775

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error"))) # 388.46066436038353

cart_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV

cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X,y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error"))) # 330.1964109339104

#####################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
#####################################

# breast_cancer veriseti

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/breast_cancer.csv")

y = df["diagnosis"]
X= df.drop(["diagnosis", "id"], axis=1)

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns = ["PC1", "PC2"])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis = 1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.set_title(f"{target.capitalize()}", fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(list(mcolors.CSS4_COLORS.values()), len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, "PC1"], dataframe.loc[indices, "PC2"], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")

# iris veriseti

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X= df.drop(["species"], axis=1) # kategorik değişkenlerin olmaması lazım

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")

# diabetes veriseti

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/8_Unsupervised_Learning/dataset/diabetes.csv")

y = df["Outcome"]
X= df.drop(["Outcome"], axis=1) # kategorik değişkenlerin olmaması lazım

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")