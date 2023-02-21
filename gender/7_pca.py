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
import matplotlib.ticker as ticker

YEAR = 2020

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

# 主成分分析

pca = PCA(copy=True, n_components=None, whiten=False)
feature = pca.fit_transform(data_norm)
print(data_norm)
print("主成分\n", feature)
print(len(feature))

df = pd.DataFrame({"pca1":feature[:, 0], "id": data_origin["id"], "year": data_origin["調査年"]})
print(df)
print(df["year"].describe())
df4map = df[df['year'].isin([YEAR])]
df4map = df4map[["id", "pca1"]]
df4map = df4map.set_index("id")
print("2010年に絞る\n", df4map)

# カラーマップと値を対応させるために、normで正規化
colors = 'Oranges'
cmap = plt.get_cmap(colors)
norm = plt.Normalize(vmin=df4map.pca1.min(), vmax=df4map.pca1.max())

# fcolは、人口の値を色に変換する関数
fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()

plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
# pictureに、df1.人口.apply(fcol)を与えることで、人口ごとの色に塗る

plt.title(YEAR)
plt.imshow(picture(df4map.pca1.apply(fcol)))
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

pca1 = pd.concat([pd.DataFrame(data_origin.columns[:]), pd.DataFrame(pca.components_[0], columns=['pca'])], axis=1)
pca1_sort = pca1.sort_values('pca', ascending=False)
plt.figure(figsize=(16, 12))
plt.title("第一主成分の構成")
plt.barh(pca1_sort[0], pca1_sort['pca'])

pca2 = pd.concat([pd.DataFrame(data_origin.columns[:]), pd.DataFrame(pca.components_[1], columns=['pca'])], axis=1)
pca2_sort = pca2.sort_values('pca', ascending=False)
plt.figure(figsize=(16, 12))
plt.title("第2主成分の構成")
plt.barh(pca2_sort[0], pca2_sort['pca'])

pca3 = pd.concat([pd.DataFrame(data_origin.columns[:]), pd.DataFrame(pca.components_[2], columns=['pca'])], axis=1)
pca3_sort = pca3.sort_values('pca', ascending=False)
plt.figure(figsize=(16, 12))
plt.title("第3主成分の構成")
plt.barh(pca3_sort[0], pca3_sort['pca'])

print("第一主成分が正\n", data_origin.columns[:][pca.components_[0] > 0])
print("第一主成分が負\n", data_origin.columns[:][pca.components_[0] < 0])
"""

