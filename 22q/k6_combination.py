import codecs
import itertools
import math
import japanize_matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMRegressor


# Load data from CSV file
from sklearn.model_selection import cross_val_score

with codecs.open("multimobidity.csv", "r", "Shift-JIS", "ignore") as file1:
    data = pd.read_table(file1, delimiter=",")
    data = data.drop('その他の疾患', axis=1)
print(data)

# Extract symptoms and anxiety scores
symptoms = data.iloc[:, 3:-1]  # Extract all columns except the last one
#symptoms = symptoms[["先天性心疾患", "免疫系疾患・アレルギー疾患", "内分泌系疾患", "消化器系疾患", "耳鼻咽喉科・顔面口腔外科系疾患"]].copy()

anxiety = data.iloc[:, -1]  # Extract the last column


"""
# Create a dictionary to store the K6 scores for each symptom combination
k6_scores = {}

# Loop over all possible symptom combinations
print("scores")
for i in range(1, len(symptoms.columns) + 1):
    for combination in itertools.combinations(symptoms.columns, i):
        # print(combination)

        # Check if all symptoms in the combination have at least one participant
        if all(symptoms[symptom].sum() > 0 for symptom in combination):

            # Check if this combination is not a subset of any larger combination
            is_valid_combination = True
            for larger_combination in k6_scores.keys():
                if set(combination).issubset(set(larger_combination)):
                    is_valid_combination = False

            # If the combination is valid, calculate the K6 score for this combination
            if is_valid_combination:
                k6 = anxiety[list(symptoms[list(combination)].all(axis=1).values)].mean()
                # print(f"{combination}: {k6}")
                # Add the K6 score to the dictionary
                k6_scores[combination] = k6

# Sort the dictionary by K6 score and get the top 10 combinations
top_combinations = sorted(k6_scores.items(), key=lambda x: x[1], reverse=True)[:10]

# Print the top combinations and their K6 scores
print("top combinations")
for combination, score in top_combinations:
    print(f"{combination}: {score}")

# Calculate the Shapley value for each symptom
n = len(symptoms.columns)
shapley_values = {symptom: 0 for symptom in symptoms.columns}
print(shapley_values)

k6_scores[()] = 0
for i in range(1, n + 1):
    for subset in itertools.combinations(symptoms.columns, i):
        if i == 1:
            marg_contrib = k6_scores[subset]

        else:
            marg_contrib = (k6_scores[subset] - k6_scores[tuple(sorted(set(symptoms.columns) - set(subset)))]) / len(subset)

        for symptom in subset:
            shapley_values[symptom] += marg_contrib

        # Normalize the Shapley values
        for symptom in symptoms.columns:
            shapley_values[symptom] /= n

# Print the Shapley values
print("Shapley values:")
for symptom, value in shapley_values.items():
    print(f"{symptom}: {value}")
"""



# Nicest style for SHAP plots
# sns.set(style='ticks')

# Just keep numeric columns
X = data.iloc[:, 3:-1]

# Cast all columns to numeric type
X = X.apply(pd.to_numeric)
y = data.iloc[:, -1]
"""
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
cv_params = {'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
             'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
             'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
             'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
             'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
             }
param_scales = {'reg_alpha': 'log',
                'reg_lambda': 'log',
                'num_leaves': 'linear',
                'colsample_bytree': 'linear',
                'subsample': 'linear',
                'subsample_freq': 'linear',
                'min_child_samples': 'linear'
                }
from sklearn.model_selection import GridSearchCV

# 最終的なパラメータ範囲（1152通り）
cv_params = {'reg_alpha': [0.0001, 0.003, 0.1],
             'reg_lambda': [0.0001, 0.1],
             'num_leaves': [2, 3, 4, 6],
             'colsample_bytree': [0.4, 0.7, 1.0],
             'subsample': [0.4, 1.0],
             'subsample_freq': [0, 7],
             'min_child_samples': [0, 2, 5, 10]
             }
# グリッドサーチのインスタンス作成
gridcv = GridSearchCV(model, cv_params, cv=5,
                      scoring=scoring, n_jobs=-1)
# グリッドサーチ実行（学習実行）
gridcv.fit(X, y)  # , **fit_params)
# 最適パラメータの表示と保持
best_params = gridcv.best_params_
best_score = gridcv.best_score_
print(f'最適パラメータ {best_params}\nスコア {best_score}')
"""
model = LGBMRegressor(colsample_bytree=0.4, min_child_samples=10, num_leaves=2, reg_alpha=0.0001, reg_lambda=0.0001,
                      subsample=0.4, subsample_freq=7)
model.fit(X, y)

scoring = 'neg_root_mean_squared_error'  # 評価指標をMSEに指定
# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=5,
                         scoring=scoring, n_jobs=-1)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')


explainer = shap.Explainer(model, X)
shap_values = explainer(X)

'''
print(shap_values.shape)
print(shap_values.shape == X.shape)
print(type(shap_values))
print(shap_values[9])
'''

shap.plots.waterfall(shap_values[6], max_display=len(X.columns))
shap.summary_plot(shap_values=shap_values,
                  features=X,
                  feature_names=X.columns)