import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import sys
import sklearn.neighbors._base

# データ読み込み
d = pd.read_csv("female_over_male.csv", encoding='utf-8')

# 調査年の「年度」を削除しデータ型を数値（int）に
year = d.iloc[:, 1]
year_n = []
for i in range(len(year)):
    year_n.append(int(year[i][:4]))
d['調査年'] = year_n

# 欠損値をすべてNaNに揃える
# df: raw dataframe
df = d.iloc[:, 1:]
df = df.replace(np.inf, np.nan)
df = df.drop('地', axis=1)
df.to_csv('genderraw.csv', encoding='Shift-JIS')
print(df)

# NaNの個数を数え（→ プロットの欠損値%に使用）、降順にprint
# すべてNaNの列はない
df_nan = df.isnull().sum()
print(df_nan)
print(df_nan.sort_values(ascending=False))

# sklearnのバージョンによって、.baseが_.base となったことによるエラーに対処
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

# 欠損値への代入を行う部分(x)のみ抜き出し
x = df.iloc[:, 2:]
yearloc = df.iloc[:, :2]
print(yearloc)

# imputation
# missforestを行った結果をdataframeにする過程で、一旦txtを挟む必要があった
# df_miss:missforest dataframe
from missingpy import MissForest

imputer = MissForest()
x1 = imputer.fit_transform(x)
np.savetxt('gendermiss.txt', x1, delimiter=',')
x1 = np.loadtxt("gendermiss.txt", delimiter=',')
x1_df = pd.DataFrame(x1)
print(x1_df)
df_miss = pd.concat([yearloc, x1_df], axis=1)
df_miss.columns = df.columns
df_miss.to_csv('gendermissingforest.csv', encoding='Shift-JIS')
print(df_miss)

# plot
# 結果はcurrent directory 内に保存される
print(df)
print(df_miss)
d_nan = d.isnull().sum()
print(df_nan)

df_plot = pd.concat([df, df_miss.iloc[:, 2:]], axis=1)
df_plot.columns = range(len(df_plot.columns))
print(df_plot)

for i in range(0, 199, 1):
    d1 = df_plot.iloc[:, [0, 1, i + 2, i + 202]]
    print(d1)
    fig, ax = plt.subplots(sharex='all')
    for key, data in d1.groupby(d1.columns[1]):
        ax.scatter(data[d1.columns[0]], data[d1.columns[3]], c='lightgray', alpha=0.5, lw=0.5)
        ax.scatter(data[d1.columns[0]], data[d1.columns[2]], c='red', alpha=0.5, lw=0.5)
        ax.set_title(f'{df.columns[i + 3]}女男比 miss（欠損値{df_nan[i + 3] / 2162 * 100:.1f}%）')
        fig.savefig(f'fmratio_miss{df.columns[i + 3]}.png')
        plt.close(fig)
