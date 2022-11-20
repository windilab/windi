import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt

# モデルはRandom Forestを使う
from sklearn.ensemble import RandomForestRegressor

# SHAP(SHapley Additive exPlanations)
import shap

from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv("gender_gap_full.csv", delimiter=",")
# print(df.head(10))

shap.initjs()  # いくつかの可視化で必要

df = df.drop(columns=["Unnamed: 0", "location", "kanji", "調査年", "地域", "year"])
# print(df.head(15))

# 特徴量 X、アウトカム y、割り当て変数 T
y = df["incidence_ratio"]
X = df.drop(["incidence_ratio"], axis=1)
# print(X.head())
# print("ランダム化前のy:\n", y)

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
model.fit(X_train, Y_train)

# 学習済みモデルの評価
# predicted_Y_val = model.predict(X_val)
# print("model_score: ", model.score(X_val, Y_val))

# shap valueで評価
# Fits the explainer
# explainer = shap.Explainer(model.predict, X_val)
explainer = shap.TreeExplainer(model)

# Calculates the SHAP values - It takes some time
shap_values = explainer.shap_values(X_val[:100])

# print(shap_values)
# print("ランダム化前のshap value\n", shap_values.abs.mean(0).values)

j = X.columns.get_loc("非労働力人口【人】")  # カラム数を抽出
print("ランダム化前のshap value\n", np.abs(shap_values[:, j]).mean())

# yをランダム化
ls = []
for i in range(1000):
    y = y.sample(frac=1, random_state=i)
    # print("ランダム化", i+1, "回目のy:\n", y)

    Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

    # sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
    # 教師データと教師ラベルを使い、fitメソッドでモデルを学習
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    model.fit(X_train, Y_train)

    # 学習済みモデルの評価
    # predicted_Y_val = model.predict(X_val)
    # print("model_score: ", model.score(X_val, Y_val))

    # shap valueで評価
    # Fits the explainer
    # explainer = shap.Explainer(model.predict, X_val)
    # Calculates the SHAP values - It takes some time
    # shap_values = explainer(X_val[:100])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:100])  # , check_additivity=False)  # 数が少ないとSHAPの予測が不正確になるためエラーになる

    print(i + 1, "回目のランダム化のshap value\n", np.abs(shap_values[:, j]).mean())

    ls.append(np.abs(shap_values[:, j]).mean())

print(ls)
plt.hist(ls)
plt.show()

# 数値的に上下5%の値をみてみる
print(ls)
print(np.quantile(ls, [0.05, 0.95]))
