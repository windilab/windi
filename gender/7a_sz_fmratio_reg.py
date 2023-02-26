import pandas as pd
import codecs
from japanmap import picture
from japanmap import pref_names
from japanmap import pref_code
import numpy as np
import pandas as pd
import seaborn as s
import matplotlib.pyplot as plt
import umap
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  # 回帰分析のライブラリ
from windi_library import GBD_japanmap_merge

df = pd.read_csv("IHME-GBD_2019_DATA_47P_15-39.csv", delimiter=",")
# cause列==Schizophrenia かつ metric列==Rateの行を抜き出し
df=df[(df['cause']=='Schizophrenia')&(df['metric']=='Rate')]
# 必要な列のみ残す
df = df.loc[:, ['year', 'location', 'sex', 'val']]
# 都道府県のID、ローマ字表記、漢字表記を並べたデータフレームに
df = GBD_japanmap_merge(df)
# 年度が1995-2015のデータのみ選択
df = df [(1995<=df['year'])&(df['year']<=2015)]

# pivot 関数で男女それぞれのvalのコラム　→　FM ratio計算
df=df.pivot(index=['year', 'ID', 'location', 'kanji'], columns='sex', values='val')
df['fmratio']=df['Female']/df['Male']
df=df.reset_index()
print(df)

# for文で男女、都道府県ごとに線形回帰
result = []
ncol, nrow = 6, 8
fig = plt.figure(figsize=(8, 8))

for i in range(1, 48):
    d1 = df[df['ID'] == i]
    model_lr = LinearRegression()
    x = d1['year']
    X = np.array(x).reshape(-1, 1)
    y = d1['fmratio']
    Y = np.array(y).reshape(-1, 1)
    lr = model_lr.fit(X, Y)
    fmcoef = lr.coef_[0, 0]  # 数値で入れるため、行列を指定
    ax = plt.subplot2grid((nrow, ncol), (i // ncol, i % ncol))
    ax = plt.plot(X, Y, 'o', color='blue')
    ax = plt.plot(X, lr.predict(X), color='black')
    plt.title(d1['location'].iloc[1])
    plt.tight_layout()
    result.append([d1['ID'].iloc[1], fmcoef])  # japanmapに渡すため、都道府県IDと並べる

# plt.savefig('schizophrenia_fmratio_regression.png')
plt.show()

# 男性の減少率 - 女性の減少率
# 日本地図にマッピング
df_result = pd.DataFrame(result)
df_result.columns = ['ID', 'fmratio_regression']
print(df_result)
# 標準化
mean = df_result['fmratio_regression'].mean()
std = df_result['fmratio_regression'].std()
df_result["value_map"] = (df_result['fmratio_regression'] - mean) / std
print(df_result)
#df_result.to_csv("sz_fmratio_z.csv")

df_result = df_result[["ID", "value_map"]]
df_result = df_result.set_index(["ID"])
df_result.to_csv("schizo_coef_z.csv")
cmap = plt.get_cmap('Greens')
norm = plt.Normalize(vmin=df_result.value_map.min(), vmax=df_result.value_map.max())
fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()

plt.rcParams['figure.figsize'] = 10, 10
plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
plt.title('Schizophrenia Female-Male ratio')
plt.imshow(picture(df_result.value_map.apply(fcol)))
# plt.savefig('schizophrenia_fmratio_mapping.png')
plt.show()
