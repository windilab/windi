import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from multiprocessing import cpu_count

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

# Borutaを実行
rf = RandomForestRegressor(n_jobs=int(cpu_count() / 2), max_depth=7)
feat_selector = BorutaPy(rf, n_estimators='auto', two_step=False, verbose=2, random_state=42)
feat_selector.fit(X_train.values, Y_train.values)
print(X_train.columns[feat_selector.support_])

# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_val_selected = X_val.iloc[:, feat_selector.support_]
print(X_val_selected.head())

# 選択したFeatureで学習
rf2 = RandomForestRegressor(n_estimators=500,
                            random_state=42,
                            n_jobs=int(cpu_count() / 2))
rf2.fit(X_train_selected.values, Y_train.values)

predicted_Y_val_selected = rf2.predict(X_val_selected.values)
print("model_score_2: ", model.score(X_val, Y_val))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(rf2.predict, X_val_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val_selected)

shap.plots.bar(shap_values, max_display=len(X_val_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_val_selected.columns))
# or
# shap.plots.beeswarm(shap_values)

print("非労働人口のshap: ", shap_values[:, "非労働力人口【人】"].abs.mean(0).values)