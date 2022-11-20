import random
import numpy as np
import matplotlib.pyplot as plt

"""
def perm_func(g1, g2):
    # グループにまとめて、シャッフルする
    g = g1 + g2
    random.shuffle(g)
    # ランダムにもとのグループと同じ数になるよう割り当てる
    new_g1 = g[:len(g1)]
    new_g2 = g[len(g1):]

    # ランダムに作ったグループから比較値をつくる
    # 今回はCVR=比率をみている
    diff = float(sum(new_g1) / len(new_g1)) - float(sum(new_g2) / len(new_g2))

    return diff


# データ:対象なのか、郡内の数が同じかは気にしないでいい
g1 = [1, 1, 1, 0, 0]
g2 = [0, 1, 0, 1]
# 1000サンプリングしてみる
res = []
for i in range(1000):
    res.append(perm_func(g1, g2))

# 実際の比率の差
data_diff = float(sum(g1) / len(g1)) - float(sum(g2) / len(g2))

# 視覚的にみてみる
# plt.hist(res)
# plt.axvline(data_diff, color='r')
# plt.show()

# 数値的に上下5%の値をみてみる
print(np.quantile(res, [0.95, 0.05]))
"""

import shap
import xgboost

# get a dataset on income prediction
X,y = shap.datasets.adult()

# train an XGBoost model (but any other model type would also work)
model = xgboost.XGBClassifier()
model.fit(X, y);

print("X\n", X)

# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, X)
shap_values = explainer(X[:100])

# get just the explanations for the positive class
shap_values = shap_values[...,1]

print("shap_values\n", shap_values)
# sex = ["Women" if shap_values[i,"Sex"].data == 0 else "Men" for i in range(shap_values.shape[0])]

#print("Featureのshap_value\n", shap_values[:,"Relationship"])
#print("Featureのshap_value?\n", shap_values[:,"Relationship"].values)
#print("Featureのshap_valueの平均?\n", shap_values[:,"Relationship"].values.mean())

print(shap_values[:,"Relationship"].abs.mean(0).values)

# shap.plots.bar(shap_values.cohorts(sex).abs.mean(0))
shap.plots.bar(shap_values)
# print(shap_values.cohorts(2).abs.mean(0))

