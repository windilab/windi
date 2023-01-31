import math

import numpy as np
import pandas as pd
import os
import csv
from scipy import stats
import matplotlib.pylab as plt
from windi_library import GBD_caliculator_kaiki, GBD_japanmap_merge
import scipy.stats as st


def wilcoxon(df):
    A_list = df["incidence_z"]
    A = list(A_list)
    print(A)
    B_list = df["social_z"]
    B = list(B_list)
    print(B)
    df = df.set_index(["location"])

    df0 = df[["incidence_z", "social_z"]]
    df0.plot(kind="bar")

    plt.figure()
    plt.scatter(A, B)
    plt.xlabel('incidence_z')
    plt.ylabel('social_z')
    plt.show()

    return stats.wilcoxon(A, B, alternative='two-sided').pvalue


def corr_CI(a, b, alpha=0.95):
    r = stats.spearmanr(a, b)[0]
    n = len(a)
    if n <= 3:
        AssertionError("Not enough amount data")
    z = 0.5 * np.log((1 + r) / (1 - r))
    za = stats.norm.ppf(0.5 + 0.5 * alpha)
    zl = z - za * math.sqrt(1 / (n - 3))
    zu = z + za * math.sqrt(1 / (n - 3))
    rhol = (math.exp(2 * zl) - 1) / (math.exp(2 * zl) + 1)
    rhou = (math.exp(2 * zu) - 1) / (math.exp(2 * zu) + 1)
    return rhol, rhou


df = pd.read_csv("data47P_15to39.csv", delimiter=",")
df = df[df["cause"] == "Schizophrenia"]
print(df)
df2 = GBD_japanmap_merge(df)
print(df2)
df_reg = GBD_caliculator_kaiki(df2)
df_reg = pd.merge(df_reg, df2, on='ID')
df_reg = df_reg[["ID", "location", "value_map"]]
df_reg = df_reg.rename(columns={'value_map': 'incidence_z'})
df_reg = df_reg.drop_duplicates()
print(df_reg)

# 社会因子のz値のマージする
df_social = pd.read_csv("social_factor_z.csv", delimiter=",")
df_social = df_social.rename(columns={'value_map': 'social_z'})
# df_social["social_z"] = -df_social["social_z"]
df_z_value = pd.merge(df_reg, df_social, left_on=['location'], right_on=['location'])
print(df_z_value)

print("ウィルコクソン符号付き順位検定", wilcoxon(df_z_value))

A_list = df_z_value["incidence_z"]
A = list(A_list)
print(A)
B_list = df_z_value["social_z"]
B = list(B_list)
print(B)

per, pep = st.pearsonr(A, B)
spr, spp = st.spearmanr(A, B)

print(per)
print(pep)
print(spr)
print(spp)

# データフレームから直接相関係数を求めることもできる
print("ピアソン相関係数:", (df_z_value[["incidence_z", "social_z"]]).corr("pearson"))
print("スピアマン相関係数", (df_z_value[["incidence_z", "social_z"]]).corr("spearman"))

print("スピアマン相関係数の95%信頼区間\n", corr_CI(A, B))