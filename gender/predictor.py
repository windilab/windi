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

# with codecs.open("gender_gap_full.csv", "r", "Shift-JIS", "ignore") as file:
#    df = pd.read_table(file, delimiter=",")

df = pd.read_csv("gender_gap_full.csv", delimiter=",")
print(df.head(10))

shap.initjs()  # いくつかの可視化で必要

df = df.drop(columns=["Unnamed: 0", "kanji", "調査年", "地域"])
df = df.set_index(["location", "year"])
print(df.head(15))

# 特徴量 X、アウトカム y、割り当て変数 T
y = df["incidence_ratio"]
X = df.drop(["incidence_ratio"], axis=1)
print(X.head())
print(y)

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(max_depth=None,
                              max_features=100,
                              # X_train.shape[1],  # The number of features to consider when looking for the best split
                              # 'sqrt'も可能
                              min_samples_split=5,
                              min_samples_leaf=1,
                              n_estimators=500,
                              # n_jobs=-1,  # number of jobs to run in parallel(-1 means using all processors)
                              random_state=2525)
model.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model.predict(X_val)
print("model_score: ", model.score(X_val, Y_val))

"""
# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_features': [i for i in range(1, X_train.shape[1], 10)],
    'random_state': [2525],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20]
    # 'max_depth': [20, 30, 40]
    # RandomForestRegressor(max_features=50, n_estimators=500,
    #                       random_state=2525)
}

gsr = GridSearchCV(
    RandomForestRegressor(),
    search_params,
    cv=3,
    n_jobs=-1,
    verbose=True
)

gsr.fit(X_train, Y_train)

# 最もよかったモデル
print(gsr.best_estimator_)
print("最もよかったモデルの評価", gsr.best_estimator_.score(X_val, Y_val))

"""
# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(model.predict, X_val)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val)

shap.plots.bar(shap_values, max_display=10)
shap.summary_plot(shap_values, max_display=10)
plt.savefig("/Volumes/Pegasus32R8/WINDI/202211/shapB.svg")
# or
# shap.plots.beeswarm(shap_values)

print("非労働人口のshap: ", shap_values[:, "非労働力人口【人】"].abs.mean(0).values)

'''
# TreeExplainerで計算（やや早い）
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val[:100])  # , check_additivity=False)  # 数が少ないとSHAPの予測が不正確になるためエラーになる

# shap.initjs()
# i = 42
# shap.force_plot(explainer.expected_value, shap_values[i], X.loc[[i]])

shap.summary_plot(shap_values, X_val[:100], plot_type="bar")
shap.summary_plot(shap_values, X_val[:100])
'''

print("非労働力人口のshap value\n", shap_values)
j = X.columns.get_loc("非労働力人口【人】")  # カラム数を抽出
print("非労働力人口【人】の列名: ", j)
print("非労働力人口のshap value絶対値の平均\n", np.abs(shap_values[:, j]).mean())
