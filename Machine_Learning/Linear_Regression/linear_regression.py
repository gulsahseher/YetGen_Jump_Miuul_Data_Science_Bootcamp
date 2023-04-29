# #####################################################
# Sales Prediction with Linear Regression
# #####################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# #####################################################
# Simple Linear Regression with OLS Using Scikit-Learn
# #####################################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/Linear_Regression/datasets/Advertising.csv")

X = df[["TV"]]
y = df[["sales"]]

#####################################
# Model
#####################################

reg_model = LinearRegression().fit(X,y)

# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]

#####################################
# Tahmin
#####################################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
# y_hat = b + w*TV = 7.032593549127693 + 150*0.047536640433019764 = 14.163089614080658

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s":9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

#####################################
# Tahmin Başarısı
#####################################

# Bütün hata metriklerinin değerleri kendi içerisindeki değişikliklere bakılarak yapılır
# MAE'nin daha düşük çıkması daha doğru olduğunu göstermez.

# MSE Değeri
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred) # 10.512652915656757
y.mean()
y.std()

# mse değerinin büyüklüğü neye göre değerlendirmemiz gerektiği düşünüldüğünde
# ortamaya ve standart sapmaya bakıyoruz
# Bu örnekte ortalamaya ve standart sapmaya bakıldığında hatanın yüksek çıktığı yorumunu
# yapabiliriz.

# RMSE
np.sqrt(mean_squared_error(y, y_pred)) # 3.2423221486546887

# MAE
mean_absolute_error(y, y_pred) # 2.549806038927486

# R-Kare
reg_model.score(X,y) # 0.611875050850071

# TV değişkeninin satış değişkenindeki değişikliği açıklama yüzdesini verir.

# #####################################################
# Multiple Linear Regression (Çoklu Doğrusal Regresyon Modeli)
# #####################################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/Linear_Regression/datasets/Advertising.csv")
df.head()
X = df.drop("sales", axis=1)
# sales'i dışarıda bırakıyoruz çünkü df içerisinde bağımsız değişkenler vardı
y = df[["sales"]]
# bağımlı değişkeni seçmek için de df içerisinden sales'i aldık ve y'ye atadık

#####################################
# Model
#####################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1) # Bağımsız değişkenlerle bağımlı değişkenleri ayrı ayrı ister.

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_[0]

reg_model.coef_

#####################################
# Tahmin
#####################################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio : 10
# newspaper : 40

# y_hat = b + w1*TV + w2*radio+ w3*newspaper = 2.9079470208164295 + 30*0.0468431 + 10*0.17854434 + 40*0.00258619 = 6.20

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)[0][0]

#####################################
# Tahmin Başarısı
#####################################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # 1.736902590147092

# Train R-Kare
reg_model.score(X_train,y_train) # 0.8959372632325174
# Yeni değişkenler eklendiğinde açıklama oranı yani başarı yükselmiştir diyebiliriz.
# Çünkü Simple Linear Regression bölümünde R-Kare 0.61 hesaplanmıştı.

# Test RMSE
y_pred = reg_model.predict(X_test) # test setinin bağımsız değişkenleri
np.sqrt(mean_squared_error(y_test, y_pred)) # 1.4113417558581582

# Test hatasının train hatasından daha yüksek çıkması beklenir.

# Test R-Kare
reg_model.score(X_test,y_test) # 0.8927605914615384

# 10 katlı Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # 1.6913531708051792

# 5 katlı Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error"))) # 1.7175247278732084

# Veriseti küçük olduğu için k-katlı Cross Validation RMSE değeri daha güvenilir bir sonuç verir diyebiliriz.

# #####################################################
# Simple Linear Regression with Gradient Descent from Scratch
# #####################################################

# Cost function MSE

def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# Update Weights

def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)

    return new_b, new_w

# Train function

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter = {:d}   b = {:.2f}   w = {:.4f}   mse = {:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))

    return cost_history, b, w

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/Linear_Regression/datasets/Advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters

learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate,num_iters)