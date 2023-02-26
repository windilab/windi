import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables

df = pd.read_csv("gender_gap_full.csv", delimiter=",")
print(df.head(10))

df = df.drop(columns=["Unnamed: 0", "location", "kanji", "調査年", "地域"])  # 都道府県のID
# df = df.set_index(["year"])
print(df.head(15))

# 特徴量 X、アウトカム y、割り当て変数 T
y = df["incidence_ratio"]
X = df.drop(["incidence_ratio"], axis=1)
print(X.head())
print(y)

# 分散が０の変数削除
del_num1 = np.where(X.var() == 0)
X = X.drop(X.columns[del_num1], axis=1)

# 変数選択（互いに相関関係にある変数の内一方を削除）
threshold_of_r = 0.7  # 変数選択するときの相関係数の絶対値の閾値(★要検討)
corr_var = search_highly_correlated_variables(X, threshold_of_r)
X.drop(X.columns[corr_var], axis=1, inplace=True)

# 同じ値を多くもつ変数の削除
inner_fold_number = 2  # CVでの分割数（予定）
rate_of_same_value = []
threshold_of_rate_of_same_value = (inner_fold_number - 1) / inner_fold_number
for col in X.columns:
    same_value_number = X[col].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / X.shape[0]))
del_var_num = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
X.drop(X.columns[del_var_num], axis=1, inplace=True)

print("残った社会指標", X.columns)

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=.3)

# モデル構築　
model = LinearRegression()

# 学習
model.fit(X_train, Y_train)

# 回帰係数
coef = pd.DataFrame({"社会指標": X.columns, "coefficient": model.coef_}).sort_values(by='coefficient')

# 結果
print("【回帰係数】\n", coef)
print("【切片】:", model.intercept_)
print("【決定係数(訓練)】:", model.score(X_train, Y_train))
print("【決定係数(テスト)】:", model.score(X_test, Y_test))
coef.to_csv("coef_gender.csv")
