import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import japanize_matplotlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from boruta import BorutaPy
from multiprocessing import cpu_count
import math

# ---------------------------
# データの読み込みと前処理
# ---------------------------
df_ir = pd.read_csv("data47P_15to39.csv", delimiter=",")
df_ir = df_ir[["kanji", "population_density", "location", "sex", "cause", "year", "val", "ID"]]
df_ir = df_ir[df_ir.cause == "Schizophrenia"]
df_ir = df_ir.set_index(["kanji", "year"])
df_ir["incidence_ratio"] = df_ir[df_ir.sex == "Female"]["val"] / df_ir[df_ir.sex == "Male"]["val"]
df_ir = df_ir.reset_index()
df_ir = df_ir[["kanji", "location", "year", "incidence_ratio", "ID"]]
df_ir = df_ir.drop_duplicates()
print("発症率")
print(df_ir.head)

df_socio = pd.read_csv("genderraw_utf.csv", delimiter=",")
print("社会環境因子")
print(df_socio)
df2 = pd.merge(df_ir, df_socio, left_on=['kanji', 'year'], right_on=['地域', 'year'])
print("merge")
print(df2)

# 不要な列を削除し、インデックスを設定
df = df2.drop(columns=["Unnamed: 0", "kanji", "地域", "ID"])
df = df[df['year'] % 5 == 0]
df = df.loc[:, df.notna().all()]
df = df.set_index(["location", "year"])
print("5の倍数年度　かつ　欠損値がない項目")
print(df.head())
df.to_csv("five_year_table.csv")

df["incidence_ratio"] = df["incidence_ratio"] * 100  # %表示にしてみる

"""
df = df[[
    "総人口【人】", "日本人人口【人】", "死亡数【人】", "平均婚姻年齢（初婚の夫）【歳婚姻年齢（初婚の夫）【歳】",
    "幼稚園教員数【人】", "小学校教員数【人】", "中学校教員数【人】", "中学校卒業者のうち進学者数【人】", "高等学校教員数【人】",
    "高等学校生徒数【人】", "高等学校卒業者のうち進学者数【人】", "高等学校卒業者のうち就職者数【人】", "大学学生数【人】",
    "所定内実労働時間数（〜2019）【時間】",
    "超過実労働時間数（〜2019）【時間】", "きまって支給する現金給与額（〜2019）【千円】", "所定内給与額（〜2019）【千円】",
    "新規学卒者初任給（高校）【千円】", "身長（中学2年）【ｃｍ】", "身長（高校2年）【ｃｍ】", "体重（小学5年）【ｋｇ】",
    "体重（中学2年）【ｋｇ】", "体重（高校2年）【ｋｇ】",
    "incidence_ratio",
]]
"""

# 回帰ターゲットとしてincidence_ratioを利用
y = df["incidence_ratio"]
X = df.drop(["incidence_ratio"], axis=1)
print("X")
print(X.head())
print('y')
print(y.head())

# 訓練データと検証データに分割（再現性のためrandom_stateを設定）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# GridSearchCVによるハイパーパラメータ探索
# ---------------------------
# 指定のハイパーパラメータグリッド
param_grid = {
    'criterion': ['squared_error', 'absolute_error'],
    'n_estimators': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'max_depth': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
}

# RandomForestRegressorのインスタンス作成
regr = RandomForestRegressor(n_jobs=int(cpu_count() / 2), random_state=42)

# GridSearchCVの設定（5-fold CV、スコアはR2）
grid_search = GridSearchCV(estimator=regr,
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           n_jobs=-1,
                           verbose=1,
                           )
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV R2 Score:", grid_search.best_score_)

# ---------------------------
# 最適なモデルの学習と評価（検証データ）
# ---------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)

r2 = r2_score(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)

print("=== 検証データでの評価 ===")
print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# -----------------------------------------------------
# 交差検証による詳細な評価（複数の指標を同時に計算）
# -----------------------------------------------------
scoring = {
    'r2': 'r2',
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error'
}
cv_results = cross_validate(best_model, X_train, y_train, cv=5, scoring=scoring)
print("=== 交差検証による評価 ===")
print("Mean R2:", np.mean(cv_results['test_r2']))
print("Mean MSE:", -np.mean(cv_results['test_neg_mean_squared_error']))
print("Mean MAE:", -np.mean(cv_results['test_neg_mean_absolute_error']))

# ---------------------------
# Borutaによる特徴選択
# ---------------------------
feat_selector = BorutaPy(
    best_model,
    n_estimators='auto',
    perc=100,
    two_step=False,
    max_iter=200,
    verbose=1,
    alpha=0.05,  # / len(X.columns),
    random_state=42
)
feat_selector.fit(X_train.values, y_train.values)
selected_features = X_train.columns[feat_selector.support_]
print("選択された特徴量:", selected_features)

# 選択された特徴量のみのデータセットを作成
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]

# ---------------------------
# 選択後のモデル再学習と評価
# ---------------------------
best_model.fit(X_train_selected, y_train)
y_pred_selected = best_model.predict(X_val_selected)

r2_selected = r2_score(y_val, y_pred_selected)
mse_selected = mean_squared_error(y_val, y_pred_selected)
rmse_selected = math.sqrt(mse_selected)
mae_selected = mean_absolute_error(y_val, y_pred_selected)

print("=== 選択後の検証データでの評価 ===")
print("R2:", r2_selected)
print("MSE:", mse_selected)
print("RMSE:", rmse_selected)
print("MAE:", mae_selected)

cv_results_selected = cross_validate(best_model, X_train_selected, y_train, cv=5, scoring=scoring)
print("=== 交差検証による評価 ===")
print("Mean R2:", np.mean(cv_results_selected['test_r2']))
print("Mean MSE:", -np.mean(cv_results_selected['test_neg_mean_squared_error']))
print("Mean MAE:", -np.mean(cv_results_selected['test_neg_mean_absolute_error']))

# ---------------------------
# SHAPによるモデル解釈
# ---------------------------
# SHAPのExplainerを利用して各特徴量の寄与度を評価
explainer = shap.Explainer(best_model.predict, X_val_selected)
shap_values = explainer(X_val_selected)

# 特定の特徴量の平均絶対SHAP値
for feature_name in ["所定内給与額（～2019）【千円】", "非労働力人口【人】", "きまって支給する現金給与額（～2019）【千円】"]:
    shap_abs_mean = np.abs(shap_values.values[:, X_val_selected.columns.get_loc(feature_name)]).mean()
    print(f"{feature_name}の平均絶対SHAP値:", shap_abs_mean)

# 各特徴量の平均絶対SHAP値を棒グラフでプロット
shap.plots.bar(shap_values, max_display=len(X_val_selected.columns))
# 詳細なSHAPサマリープロット
shap.summary_plot(shap_values, max_display=len(X_val_selected.columns))
