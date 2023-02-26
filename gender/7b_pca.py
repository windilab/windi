import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from japanmap import pref_names
from japanmap import pref_code
from japanmap import picture
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# GBDデータから男女比の減少率のデータフレームを計算する
def calc_reg(df2):
    df2 = df2[["id", "year", "pc1"]]
    # df2 = df2[df2.year.isin(YEARS)]  # 年を指定
    print("計算前のデータフレーム\n", df2)

    df2 = df2.set_index("id")
    # 回帰係数を計算
    model_lr = LinearRegression()
    coef = []

    for n in df2.index.values:  # 都道府県名毎に回帰係数を求める
        df3 = pd.DataFrame(df2.loc[n]).reset_index()
        # print("reset_index\n", df3)
        x = df3[["year"]]
        y = df3[["pc1"]]
        model_lr.fit(x, y)
        coef = coef + [[n, model_lr.coef_[0, 0]]]

    df_coef = pd.DataFrame(coef, columns=['id', 'coef'])
    print(df_coef)

    # 標準化
    mean = df_coef["coef"].mean()
    std = df_coef["coef"].std()
    df_coef["value_map"] = (df_coef["coef"] - mean) / std

    # for文で都道府県ごとに線形回帰
    result = []
    ncol, nrow = 6, 8
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, 48):
        d1 = df[df['id'] == i]
        x = d1['year']
        X = np.array(x).reshape(-1, 1)
        y = d1['pc1']
        Y = np.array(y).reshape(-1, 1)
        mlr = model_lr.fit(X, Y)
        coef = mlr.coef_[0, 0]  # 数値で入れるため、行列を指定
        ax = plt.subplot2grid((nrow, ncol), (i // ncol, i % ncol))
        ax = plt.plot(X, Y, 'o', color='red')
        ax = plt.plot(X, mlr.predict(X), color='black')
        plt.title(d1['location'].iloc[1])
        plt.tight_layout()
    plt.show()
    return df_coef

# YEAR = 2020

data_origin = pd.read_csv("gendergap_missforest_boruta_extract.csv", delimiter=",", encoding='utf-8')
print(data_origin)

data_origin['id'] = data_origin['地域']
for i in range(len(data_origin)):
    data_origin['id'][i] = pref_code(data_origin['地域'][i])

print(data_origin)

# 標準化
sc = preprocessing.StandardScaler()
data_norm = sc.fit_transform(data_origin.drop('id', axis=1).drop('地域', axis=1).drop('調査年', axis=1))

# colを作る
# col_years
col_years = (np.floor(np.arange(0, 47 * 46, 1) / 47) + 1) / 46

"""
col_years
# 東京は27番目、大阪は10番目
col_city = np.zeros(47 * 46)
for i in range(0, 47 * 46):
    if i % 47 == 27:
        col_city[i] = 1
    else:
        if i % 47 == 10:
            col_city[i] = 1
        else:
            col_city[i] = 0.2

# col_2000
col_2000 = (np.floor(np.arange(0, 47 * 46, 1) / 47) + 1) / 92
for i in range(0, 47 * 46):
    if i // 47 == 25:
        col_2000[i] = 1
    else:
        if i // 47 == 15:
            col_2000[i] = 0.8
"""

# 主成分分析
pca = PCA(copy=True, n_components=None, whiten=False)
feature = pca.fit_transform(data_norm)
print(data_norm)
print("主成分\n", feature)
print(len(feature))

df = pd.DataFrame({"pc1": feature[:, 0], "id": data_origin["id"], "year": data_origin["調査年"],
                   "location": data_origin["地域"]})
print(df)
# print(df["year"].describe())
df = df[df['year'].isin(range(1995, 2016))]
df4map = calc_reg(df)
df4map = df4map[["id", "value_map"]]
df4map = df4map.set_index("id")
print("1995年から2015年に絞る\n", df4map.head(47))
df4map.to_csv("pc1_coef_z.csv")

# カラーマップと値を対応させるために、normで正規化
colors = 'Oranges'
cmap = plt.get_cmap(colors)
norm = plt.Normalize(vmin=df4map.value_map.min(), vmax=df4map.value_map.max())

# fcolは、人口の値を色に変換する関数
fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()

plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
# pictureに、df1.人口.apply(fcol)を与えることで、人口ごとの色に塗る

# plt.title(YEAR)
plt.imshow(picture(df4map.value_map.apply(fcol)))
plt.show()

"""
# 第一主成分と第二種成分での弁別
plt.figure(figsize=(8, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.4, c=col_years, cmap='winter')
plt.grid()
plt.title("年度で色分け　青:1975 緑:2020")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.colorbar()
plt.show()

# 第一主成分と第二種成分での弁別
plt.figure(figsize=(8, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.6, c=col_city, cmap='Wistia')
plt.grid()
plt.title("東京・大阪を色分け")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.colorbar()
plt.show()

# 第一主成分と第二種成分での弁別
plt.figure(figsize=(8, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.4, c=col_2000, cmap='cool')
plt.grid()
plt.title("1990, 2000年を色分け")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.colorbar()
plt.show()

# 第一主成分と第二種成分での弁別
plt.figure(figsize=(8, 6))
plt.scatter(feature[:, 0], feature[:, 2], alpha=0.4)
plt.grid()
plt.title(PCA)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# 第一、第二主成分にどの程度各項目が影響を与えたか
plt.figure(figsize=(20, 20))
for x, y, name in zip(pca.components_[0], pca.components_[1], data_origin.columns[:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("contribution of items")
plt.show()

# 主成分分析の累積寄与率のプロット
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()

plt.figure(figsize=(16, 12))
plt.title("第一主成分の構成")
plt.barh(data_origin.columns[:], pca.components_[0])

pc1 = pd.concat([pd.DataFrame(data_origin.columns[:]), pd.DataFrame(pca.components_[0], columns=['pca'])], axis=1)
pc1_sort = pc1.sort_values('pca', ascending=False)
plt.figure(figsize=(16, 12))
plt.title("第一主成分の構成")
plt.barh(pc1_sort[0], pc1_sort['pca'])

pc2 = pd.concat([pd.DataFrame(data_origin.columns[:]), pd.DataFrame(pca.components_[1], columns=['pca'])], axis=1)
pc2_sort = pc2.sort_values('pca', ascending=False)
plt.figure(figsize=(16, 12))
plt.title("第2主成分の構成")
plt.barh(pc2_sort[0], pc2_sort['pca'])

pc3 = pd.concat([pd.DataFrame(data_origin.columns[:]), pd.DataFrame(pca.components_[2], columns=['pca'])], axis=1)
pc3_sort = pc3.sort_values('pca', ascending=False)
plt.figure(figsize=(16, 12))
plt.title("第3主成分の構成")
plt.barh(pc3_sort[0], pc3_sort['pca'])

print("第一主成分が正\n", data_origin.columns[:][pca.components_[0] > 0])
print("第一主成分が負\n", data_origin.columns[:][pca.components_[0] < 0])
"""

