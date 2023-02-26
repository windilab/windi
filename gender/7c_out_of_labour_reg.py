import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from japanmap import picture, pref_names
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from windi_library import GBD_japanmap_merge

YEARS = range(1995, 2016)  # 年を指定
# データ読み込み
df = pd.read_csv("estat_rawdata.csv", encoding='Shift_JIS')

# 調査年の「年度」を削除しデータ型を数値（int）に
year = df.loc[:, '調査年']
year_n = []
for i in range(len(year)):
    year_n.append(int(year[i][:4]))
df['調査年'] = year_n

# 列名のコードを削除
colnames = []
target = '_'
for i in range(len(df.columns)):
    idx = df.columns[i].find(target)
    r = df.columns[i][idx + 1:]
    colnames.append(r)

df.columns = colnames

# プロットに必要なデータのみ抜き出し（ここでは調査年、地域、男女の非労働力人口）
# 非労働力人口以外の社会指標をプロットする場合はここを変える
df = df.loc[:, ['調査年', '地域', '非労働力人口（男）【人】', '非労働力人口（女）【人】']]
df.columns = ['year', 'kanji', 'male', 'female']  # GBD_japanmap_mergeを使うためにカラム名をkanjiに
df = df.dropna()
df = df[df['year'].isin(YEARS)]
# 都道府県のID、ローマ字表記、漢字表記を並べたデータフレームに
df = GBD_japanmap_merge(df)

print(df)

# for文で男女、都道府県ごとに線形回帰
result = []
ncol, nrow = 6, 8
fig = plt.figure(figsize=(8, 8))

for i in range(1, 48):
    d1 = df[df['ID'] == i]
    # print(d1)
    model_lr = LinearRegression()
    x = d1['year']
    X = np.array(x).reshape(-1, 1)
    y = d1['male']
    Y = np.array(y).reshape(-1, 1)
    z = d1['female']
    Z = np.array(z).reshape(-1, 1)
    mlr = model_lr.fit(X, Y)
    mcoef = mlr.coef_[0, 0]  # 数値で入れるため、行列を指定
    ax = plt.subplot2grid((nrow, ncol), (i // ncol, i % ncol))
    ax = plt.plot(X, Y, 'o', color='blue')
    ax = plt.plot(X, mlr.predict(X), color='black')
    model_lr = LinearRegression()
    flr = model_lr.fit(X, Z)
    fcoef = flr.coef_[0, 0]  # 数値で入れるため、行列を指定
    ax = plt.plot(X, Z, 'o', color='red')
    ax = plt.plot(X, flr.predict(X), color='black')
    plt.title(d1['location'].iloc[1])
    plt.tight_layout()
    result.append([d1['ID'].iloc[1], mcoef, fcoef])  # japanmapに渡すため、都道府県IDと並べる

# plt.savefig(r"C:\Users\piinb\Documents\DR\WIND\socialfactor_regression.png")
plt.show()

# 男性の減少率 - 女性の減少率
# 日本地図にマッピング
df_result = pd.DataFrame(result)
df_result.columns = ['ID', 'male_regression', 'female_regression']
print(df_result)
df_result['male_female'] = df_result['male_regression'] - df_result['female_regression']
mean = df_result["male_female"].mean()
std = df_result["male_female"].std()
# 標準化
df_result["value_map"] = (df_result["male_female"] - mean) / std
print(df_result)
df_result.to_csv("social_factor_z.csv")

df_result = df_result[["ID", "value_map"]]
df_result = df_result.set_index(["ID"])
cmap = plt.get_cmap('Blues')
norm = plt.Normalize(vmin=df_result.value_map.min(), vmax=df_result.value_map.max())
fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()

plt.rcParams['figure.figsize'] = 10, 10
plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
plt.title('非労働力人口　男性回帰係数 - 女性回帰係数 のz値')
plt.imshow(picture(df_result.value_map.apply(fcol)))
# plt.savefig(r'C:\Users\piinb\Documents\DR\WIND\socialfactor_mapping.png')
plt.show()
