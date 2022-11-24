import pandas as pd
import numpy as np
import japanize_matplotlib
import seaborn as s
import matplotlib.pyplot as plt

# モデルはRandom Forestを使う
from sklearn.ensemble import RandomForestRegressor

# SHAP(SHapley Additive exPlanations)
import shap

from sklearn.model_selection import train_test_split, GridSearchCV

TRIAL = 1000  # ランダム化を何回するか
FEATURE = "非労働力人口【人】"  # 調べたい項目

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

j = X.columns.get_loc(FEATURE)  # カラム数を抽出
ls0 = []
for i in range(TRIAL):
    model = RandomForestRegressor(random_state=i)

    model.fit(X_train, Y_train)

    # 学習済みモデルの評価
    predicted_Y_val = model.predict(X_val)
    print("model_score: ", model.score(X_val, Y_val))

    # shap valueで評価

    # 短時間の近似式
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:100])

    # print(shap_values)

    # print("shap value\n", shap_values)

    ls0.append(np.abs(shap_values[:, j]).mean())

    print("random_state ", i, ": ランダム化前のshap value近似値\n", np.abs(shap_values[:, j]).mean())


true_shap = pd.DataFrame(ls0, columns=['shap_value'])
true_shap['color'] = 1
print("ランダム化前の95%CI:", np.quantile(ls0, [0.025, 0.975]))

# print(shap_values)
# print("ランダム化前のshap value\n", shap_values.abs.mean(0).values)


print("ランダム化前のshap value\n", np.abs(shap_values[:, j]).mean())

# yをランダム化
ls = []
for i in range(TRIAL):
    y = y.sample(frac=1, random_state=i)
    # print("ランダム化", i+1, "回目のy:\n", y)

    Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

    # sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
    # 教師データと教師ラベルを使い、fitメソッドでモデルを学習
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)

    # 学習済みモデルの評価
    # predicted_Y_val = model.predict(X_val)
    print("model_score: ", model.score(X_val, Y_val))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:100])  # , check_additivity=False)  # 数が少ないとSHAPの予測が不正確になるためエラーになる

    print(i + 1, "回目のランダム化のshap value\n", np.abs(shap_values[:, j]).mean())

    ls.append(np.abs(shap_values[:, j]).mean())

# 数値的に上下5%の値をみてみる
print(ls)
print("ランダム化後の95%CI:", np.quantile(ls, [0.025, 0.975]))

shap_random = pd.DataFrame(ls, columns=['shap_value'])
shap_random['color'] = 0

# ランダム化前後のshapのデータフレームを結合
df4plot = pd.concat([shap_random, true_shap])
df4plot["shap_value"] = np.sqrt(df4plot["shap_value"])

s.displot(data=df4plot, x='shap_value', hue='color', multiple='stack')
plt.show()


