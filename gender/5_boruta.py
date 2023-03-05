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
    max_depth = trial.suggest_int('max_depth', 2, 2)
    max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
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

# shap.initjs()  # いくつかの可視化で必要

df = df.drop(columns=["Unnamed: 0", "kanji", "調査年", "地域", "ID"])
df = df.set_index(["location", "year"])
print(df.head(15))

# 特徴量 X、アウトカム y、割り当て変数 T
y = df["incidence_ratio"]
X = df.drop(["incidence_ratio"], axis=1)
print(X.head())
print(y)

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)


# optunaで最適化
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


# チューニングしたハイパーパラメーターをフィット

optimised_rf = RandomForestRegressor(bootstrap=study.best_params['bootstrap'],
                                     criterion=study.best_params['criterion'],
                                     max_depth=study.best_params['max_depth'],
                                     max_features=study.best_params['max_features'],
                                     max_leaf_nodes=study.best_params['max_leaf_nodes'],
                                     n_estimators=study.best_params['n_estimators'],
                                     min_samples_split=study.best_params['min_samples_split'],
                                     min_samples_leaf=study.best_params['min_samples_leaf'],
                                     n_jobs=int(cpu_count() / 2))
"""
optimised_rf = RandomForestRegressor(criterion='squared_error',
                                     bootstrap='False',
                                     max_depth=2,
                                     max_features=1.0,
                                     max_leaf_nodes=299,
                                     n_estimators=590,
                                     min_samples_split=2,
                                     min_samples_leaf=5,
                                     n_jobs=int(cpu_count() / 2))
"""
optimised_rf.fit(X_train, Y_train)

"""best_param:{'criterion': 'squared_error', 'bootstrap': 'False', 'max_depth': 2, 'max_features': 1.0, 
'max_leaf_nodes': 299, 'n_estimators': 590, 'min_samples_split': 2, 'min_samples_leaf': 5} 
==================== 
best_value:0.6691932863535266 
==================== 
"""

# 学習済みモデルの評価
predicted_Y_val = optimised_rf.predict(X_val)
print("model_score: ", optimised_rf.score(X_val, Y_val))

# Borutaを実行
feat_selector = BorutaPy(
    optimised_rf,
    n_estimators='auto',
    perc=100,
    two_step=False,
    verbose=2,
    random_state=42)
feat_selector.fit(X_train.values, Y_train.values)
print(X_train.columns[feat_selector.support_])

# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_val_selected = X_val.iloc[:, feat_selector.support_]
print(X_val_selected.head())

"""
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

"""

# チューニングしたハイパーパラメーターをフィット
optimised_rf2 = optimised_rf
optimised_rf2.fit(X_train_selected, Y_train)

predicted_Y_val_selected = optimised_rf2.predict(X_val_selected.values)
print("model_score_2: ", optimised_rf2.score(X_val_selected, Y_val))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(optimised_rf2.predict, X_val_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val_selected)

shap.plots.bar(shap_values, max_display=len(X_val_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_val_selected.columns))
# or
# shap.plots.beeswarm(shap_values)

print("非労働人口のshap: ", shap_values[:, "非労働力人口【人】"].abs.mean(0).values)
