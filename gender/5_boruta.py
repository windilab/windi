import optuna
import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from multiprocessing import cpu_count


# optunaの目的関数を設定する
def objective(trial):
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
    bootstrap = trial.suggest_categorical('bootstrap', ['True', 'False'])
    max_depth = trial.suggest_int('max_depth', 1, 1000)
    max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 1000)
    n_estimators = trial.suggest_int('n_estimators', 1, 1000)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    regr = RandomForestRegressor(bootstrap=bootstrap, criterion=criterion,
                                 max_depth=max_depth, max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 n_jobs=int(cpu_count() / 2))

    score = cross_val_score(regr, X_train, Y_train, cv=5, scoring="r2")
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean


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

# optunaで学習
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 最適化したハイパーパラメータの確認
print('check!!!')
print('best_param:{}'.format(study.best_params))
print('====================')

# 最適化後の目的関数値
print('best_value:{}'.format(study.best_value))
print('====================')

# 最適な試行
print('best_trial:{}'.format(study.best_trial))
print('====================')

# トライアルごとの結果を確認
for i in study.trials:
    print('param:{0}, eval_value:{1}'.format(i[5], i[2]))
print('====================')

# チューニングしたハイパーパラメーターをフィット
optimised_rf = RandomForestRegressor(bootstrap=study.best_params['bootstrap'], criterion=study.best_params['criterion'],
                                     max_depth=study.best_params['max_depth'],
                                     max_features=study.best_params['max_features'],
                                     max_leaf_nodes=study.best_params['max_leaf_nodes'],
                                     n_estimators=study.best_params['n_estimators'],
                                     min_samples_split=study.best_params['min_samples_split'],
                                     min_samples_leaf=study.best_params['min_samples_leaf'],
                                     n_jobs=int(cpu_count() / 2))

optimised_rf.fit(X_train, Y_train)

"""
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
"""

# 学習済みモデルの評価
predicted_Y_val = optimised_rf.predict(X_val)
print("model_score: ", optimised_rf.score(X_val, Y_val))

# Borutaを実行
rf = RandomForestRegressor(n_jobs=int(cpu_count() / 2), max_depth=7)
feat_selector = BorutaPy(rf, n_estimators='auto', two_step=False, verbose=2, random_state=42)
feat_selector.fit(X_train.values, Y_train.values)
print(X_train.columns[feat_selector.support_])

# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_val_selected = X_val.iloc[:, feat_selector.support_]
print(X_val_selected.head())

# optunaで学習
study2 = optuna.create_study(direction='maximize')
study2.optimize(objective, n_trials=100)


# 最適化したハイパーパラメータの確認
print('check!!!')
print('best_param:{}'.format(study2.best_params))
print('====================')

# 最適化後の目的関数値
print('best_value:{}'.format(study2.best_value))
print('====================')

# 最適な試行
print('best_trial:{}'.format(study2.best_trial))
print('====================')

# トライアルごとの結果を確認
for i in study2.trials:
    print('param:{0}, eval_value:{1}'.format(i[5], i[2]))
print('====================')

# チューニングしたハイパーパラメーターをフィット
optimised_rf2 = RandomForestRegressor(bootstrap=study.best_params['bootstrap'],
                                      criterion=study.best_params['criterion'],
                                      max_depth=study.best_params['max_depth'],
                                      max_features=study.best_params['max_features'],
                                      max_leaf_nodes=study.best_params['max_leaf_nodes'],
                                      n_estimators=study.best_params['n_estimators'],
                                      min_samples_split=study.best_params['min_samples_split'],
                                      min_samples_leaf=study.best_params['min_samples_leaf'],
                                      n_jobs=int(cpu_count() / 2))

optimised_rf2.fit(X_train_selected, Y_train)

# 選択したFeatureで学習
rf2 = RandomForestRegressor(bootstrap=study.best_params['bootstrap'], criterion=study.best_params['criterion'],
                            max_depth=study.best_params['max_depth'],
                            max_features=study.best_params['max_features'],
                            max_leaf_nodes=study.best_params['max_leaf_nodes'],
                            n_estimators=study.best_params['n_estimators'],
                            min_samples_split=study.best_params['min_samples_split'],
                            min_samples_leaf=study.best_params['min_samples_leaf'],
                            n_jobs=int(cpu_count() / 2))
rf2.fit(X_train_selected.values, Y_train.values)

predicted_Y_val_selected = rf2.predict(X_val_selected.values)
print("model_score_2: ", rf2.score(X_val_selected, Y_val))

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
