import pandas as pd

# データフレームの整形
df0 = pd.read_table("22qweb_dataset_dummy.csv", delimiter=",")
print(df0.head(20))
df0 = df0.fillna(0)

df_id = df0.iloc[:, :3]
print(df_id)

k = 4  # 1DataFrameあたりの列数
dfs = [df0.iloc[:, i:i+k] for i in range(3, len(df0.columns), k)]
for i, df_i in enumerate(dfs):
    print(df_i.head())
    df_i["val"] = df_i.iloc[:, 3]
    df_i["age_ctgr"] = i
    df_i["body"] = df_i.iloc[:, 0]
    df_i["intel"] = df_i.iloc[:, 1]
    df_i["heart"] = df_i.iloc[:, 2]
    df_i["dis_ctgr"] = df_i["body"]*1 + df_i["intel"]*2 + df_i["heart"]*4  # 障害の区分を0～7で
    print(df_i.head())
    df_i = df_i[["age_ctgr", "dis_ctgr", "val"]]
    print(df_i.head())
    df_i = pd.concat([df_id, df_i], axis=1, join='outer')
    print(df_i.head())
    fname = "transition_" + str(i) + ".csv"
    df_i.to_csv(fname)

df1 = pd.read_table("transition_14.csv", delimiter=",")
for i in range(14):
    file = "transition_" + str(i) + ".csv"
    df2 = pd.read_table(file, delimiter=",")
    df1 = pd.concat([df1, df2], join='outer')

print(df1.head())
df1.to_csv("transition_plot.csv")
