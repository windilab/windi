import pandas as pd
import codecs

# データフレームの整形
df0 = pd.read_table("22qweb_dataset_q7-1.csv", delimiter=",")
df0 = df0.set_index("No")
print(df0.head(20))
# df0 = df0.fillna(0)

# df_id = df0.iloc[:, :2]
# print(df_id)

k = 4  # 1DataFrameあたりの列数
list_df = pd.DataFrame(columns=["sex", "age_now", "age_ctgr", "body", "intel", "heart", "val"])
print(list_df)

dfs = [df0.iloc[:, i:i + k] for i in range(2, len(df0.columns), k)]
for i, df_i in enumerate(dfs):
    print(df_i.head())
    df_i["sex"] = df0["sex.child"].copy()
    df_i["age_now"] = df0["age.child"].copy()
    df_i["val"] = df_i.iloc[:, 3]
    df_i["age_ctgr"] = i
    df_i["body"] = df_i.iloc[:, 0]
    df_i["intel"] = df_i.iloc[:, 1]
    df_i["heart"] = df_i.iloc[:, 2]
    # df_i["dis_ctgr"] = df_i["body"]*1 + df_i["intel"]*2 + df_i["heart"]*4  # 障害の区分を0～7で
    print(df_i.head())
    df_i = df_i[["sex", "age_now", "age_ctgr", "body", "intel", "heart", "val"]]
    print(df_i.head())
    list_df = pd.concat([list_df, df_i], join='outer')

print(list_df.head())
list_df = list_df.dropna(subset=['val'])
print(list_df.head())
list_df = list_df.fillna(0)
print(list_df.head())
list_df.to_csv("transition_shapley.csv")
