# #####################################################
# KNN
# #####################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modelling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,  classification_report
pd.set_option("display.max_columns", None)

#####################################
# 1. Exploratory Data Analysis
#####################################

df = pd.read_csv("C:/Users/GULSAH/Documents/PycharmProjects/Machine_Learning/5_KNN/datasets/diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

#####################################
# 2. Data Preprocessing & Feature Engineering
#####################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

#####################################
# 3. Model & Prediction
#####################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)[0]

#####################################
# 4. Model Evaluation
#####################################

# Confussion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc: 0.83
# f1: 0.74

# AUC
roc_auc_score(y, y_prob) # 0.90

# Cross - Validation
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean() # 0.733112638994992
cv_results["test_f1"].mean() # 0.5905780011534191
cv_results["test_roc_auc"].mean() # 0.7805279524807827

# Cross-validation sonuçları daha güvenilirdir.

# Modelin başarımı arttırmak için:
# 1. Örnek boyutu arttırılabilir
# 2. Veri ön işleme yapılabilir
# 3. Özellik mühendisliği yapılabilir
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

#####################################
# 5. Hyperparameter Optimization
#####################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1, verbose=1).fit(X,y)

knn_gs_best.best_params_ # {'n_neighbors': 17}

#####################################
# 6. Final Model
#####################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean() # 0.7669892199303965
cv_results["test_f1"].mean() # 0.6170909049720137
cv_results["test_roc_auc"].mean() # 0.8127938504542278

# Final Modeli ile Tahmin
random_user = X.sample(1)
knn_final.predict(random_user)[0]