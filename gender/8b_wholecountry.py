import math

import pandas as pd
import numpy as np
import scipy.stats as stats
import codecs


def corr_CI(a, b, alpha = 0.95):
    r = stats.pearsonr(a, b)[0]
    n = len(a)
    if n <= 3:
        AssertionError("Not enough amount data")
    z= 0.5*np.log((1+r)/(1-r))
    za = stats.norm.ppf(0.5 + 0.5 * alpha)
    zl = z - za * math.sqrt(1/(n-3))
    zu = z + za * math.sqrt(1/(n-3))
    rhol = (math.exp(2 * zl) - 1 )/ (math.exp(2 * zl) +1 )
    rhou = (math.exp(2 * zu) - 1 )/ (math.exp(2 * zu) +1 )
    return r, rhol, rhou


with codecs.open("C:/Users/sawai/PycharmProjects/windi/gender/FEI_PREF_230705114019.csv", "r", "Shift-JIS",
                 "ignore") as file:
    df1 = pd.read_table(file, delimiter=",")

print(df1.head())
df1 = df1[["year", "male", "female"]]
df1 = df1.set_index("year")
print(df1)
df1["ratio_labour"] = df1["female"] / df1["male"]
print(df1)
df1 = df1[["ratio_labour"]]

df2 = pd.read_csv("IHME-GBD_2019_DATA_JPN_15-39.csv", delimiter=",")
print(df2.head())
df2 = df2[df2.metric == "Rate"]
df2 = df2[df2.cause == "Schizophrenia"]
print(df2)
df2 = df2[df2.location == "Japan"]
print(df2)
df2 = df2[["cause", "location", "year", "sex", "val"]]
print(df2)
df2 = df2[["year", "sex", "val"]]
df2 = df2.set_index("year")
df2["ratio_incidence"] = df2[df2.sex == "Female"]["val"] / df2[df2.sex == "Male"]["val"]
df2 = df2[["ratio_incidence"]]
print(df2)

df = pd.merge(df1, df2, left_index=True, right_index=True)
print(df)

# 相関係数の計算
A = df['ratio_labour']
B = df['ratio_incidence']

# 結果の表示
print("相関係数と95%信頼区間\n", corr_CI(A, B))
