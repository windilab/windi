import pandas as pd
from windi_library import analysis_47P
from windi_library import japan_all_incidence_curve
import codecs

# 都道府県別
# 関数内で、疾患を指定しているため、要確認！
df = pd.read_csv("data47P_15to39.csv", delimiter=",")
df_reg = analysis_47P(df)
df_reg = df_reg.drop_duplicates()
print("発症率の回帰係数の差のz値", df_reg[["kanji", "value_map"]])

# 全国
# japan_all_incidence_curve()

# ジェンダー指標と結合したdataframeを作る
df = pd.read_csv("data47P_15to39.csv", delimiter=",")
print(df.head(10))

df = df[["kanji", "population_density", "location", "sex", "cause", "year", "val", "ID"]]
print(df.head(10))

df = df[df.cause == "Schizophrenia"]
df = df.set_index(["kanji", "year"])
df["incidence_ratio"] = df[df.sex == "Female"]["val"] / df[df.sex == "Male"]["val"]
print(df.head(10))
df = df.reset_index()
print(df.head(10))

df0 = df[["kanji", "location", "year", "incidence_ratio", "ID"]]
print(df0.head(10))
df0 = df0.drop_duplicates()
print(df0.head(10))

with codecs.open("gendergap_missforest.csv", "r", "Shift-JIS", "ignore") as file:
    df1 = pd.read_table(file, delimiter=",")
print(df1.head(10))

df2 = pd.merge(df0, df1, left_on=['kanji', 'year'], right_on=['地域', '調査年'])
print(df2.head(10))
# df2.to_csv("gender_gap_full.csv")
