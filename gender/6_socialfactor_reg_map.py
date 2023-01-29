import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from japanmap import picture
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

pref = {"Hokkaidō": "北海道",
        "Aomori": "青森県",
        "Iwate": "岩手県",
        "Miyagi": "宮城県",
        "Akita": "秋田県",
        "Yamagata": "山形県",
        "Fukushima": "福島県",
        "Ibaraki": "茨城県",
        "Tochigi": "栃木県",
        "Gunma": "群馬県",
        "Saitama": "埼玉県",
        "Chiba": "千葉県",
        "Tōkyō": "東京都",
        "Kanagawa": "神奈川県",
        "Niigata": "新潟県",
        "Toyama": "富山県",
        "Ishikawa": "石川県",
        "Fukui": "福井県",
        "Yamanashi": "山梨県",
        "Nagano": "長野県",
        "Gifu": "岐阜県",
        "Shizuoka": "静岡県",
        "Aichi": "愛知県",
        "Mie": "三重県",
        "Shiga": "滋賀県",
        "Kyōto": "京都府",
        "Ōsaka": "大阪府",
        "Hyōgo": "兵庫県",
        "Nara": "奈良県",
        "Wakayama": "和歌山県",
        "Tottori": "鳥取県",
        "Shimane": "島根県",
        "Okayama": "岡山県",
        "Hiroshima": "広島県",
        "Yamaguchi": "山口県",
        "Tokushima": "徳島県",
        "Kagawa": "香川県",
        "Ehime": "愛媛県",
        "Kōchi": "高知県",
        "Fukuoka": "福岡県",
        "Saga": "佐賀県",
        "Nagasaki": "長崎県",
        "Kumamoto": "熊本県",
        "Ōita": "大分県",
        "Miyazaki": "宮崎県",
        "Kagoshima": "鹿児島県",
        "Okinawa": "沖縄県"}

prefecture = list(pref.keys())
print(prefecture)

# データ読み込み
df = pd.read_csv("estat_rawdata.csv", encoding='Shift_JIS')

# 調査年の「年度」を削除しデータ型を数値（int）に
year = df.loc[:, '調査年']
year_n = []
for i in range(len(year)):
    year_n.append(int(year[i][:4]))
df['調査年'] = year_n

# 都道府県名をアルファベット表記に
df = df.replace(pref.values(), pref.keys())
print(df)

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
df.columns = ['year', 'location', 'male', 'female']
df = df.dropna()

print(df)

# for文で男女、都道府県ごとに線形回帰
result = []
ncol, nrow = 6, 8
fig = plt.figure(figsize=(8, 8))

for i in range(47):
    d1 = df[df['location'] == prefecture[i]]
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
    plt.title(prefecture[i])
    plt.tight_layout()
    result.append([prefecture[i], mcoef, fcoef])

# plt.savefig(r"C:\Users\piinb\Documents\DR\WIND\socialfactor_regression.png")
plt.show()

# 男性の減少率 - 女性の減少率
# 日本地図にマッピング
df_result = pd.DataFrame(result)
df_result.columns = ['location', 'male_regression', 'female_regression']
print(df_result)
df_result['male_female'] = df_result['male_regression'] - df_result['female_regression']
mean = df_result["male_female"].mean()
std = df_result["male_female"].std()
# 標準化
df_result["value_map"] = (df_result["male_female"] - mean) / std
print(df_result)
df_result.to_csv("social_factor_z.csv")
cmap = plt.get_cmap('Blues')
norm = plt.Normalize(vmin=df_result.value_map.min(), vmax=df_result.value_map.max())
fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()

plt.rcParams['figure.figsize'] = 10, 10
plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
plt.title('非労働力人口　男性回帰係数 - 女性回帰係数 のz値')
plt.imshow(picture(df_result.value_map.apply(fcol)))
# plt.savefig(r'C:\Users\piinb\Documents\DR\WIND\socialfactor_mapping.png')
plt.show()
