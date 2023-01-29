import pandas as pd
import os
import csv
from scipy import stats
import matplotlib.pylab as plt
from windi_library import GBD_caliculator_kaiki, GBD_japanmap_merge


def wilcoxon(df):

    A_list = df["incidence_z"]
    A = list(A_list)
    print(A)
    B_list = df["social_z"]
    B = list(B_list)
    print(B)

    df0 = df[["incidence_z", "social_z"]]
    df0.plot(kind="bar")
    plt.show()

    return stats.wilcoxon(A, B, alternative='two-sided').pvalue


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
df_social["social_z"]= -df_social["social_z"]
df_z_value = pd.merge(df_reg, df_social, left_on=['location'], right_on=['location'])
print(df_z_value)

print("ウィルコクソン符号付き順位検定", wilcoxon(df_z_value))