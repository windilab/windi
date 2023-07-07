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

YEARS = range(1995, 2020)  # 年を指定


# 47都道府県別のデータフレーム、インプット
def japan_47P_incidence():
    df = pd.read_csv("IHME-GBD_2019_DATA_47P_15-39.csv", delimiter=",")
    print(df.head())

    title_age = df.age[0]

    # 40-64歳のデータを読み込んだ時のみ処理する
    if ((title_age != "All Ages") and (title_age != "15 to 39")) and (title_age != "65 to 80"):
        # title = df.cause[0] # 病名

        d = df.set_index(["year", "age", "sex", "location", "cause"])
        # print("1", d.head())

        d["population"] = d[d.metric == "Number"]["val"] / d[d.metric == "Rate"]["val"]  # 年齢別人口/10万人

        # print("2", d.head())

        d = d.reset_index()

        d = d[d.metric == "Number"]  # 年齢別発症数

        # Age 40-44, ..., 60-64を合計して一つにまとめる
        d = d[["year", "age", "sex", "val", "population", "location", "cause"]]
        d_sum = d.groupby(["sex", "year", "location", "cause"]).sum()

        print("40-64歳の発症数合計", d_sum)

        d_rate = (d_sum["val"] / d_sum["population"])  # 発症率の計算
        d_rate.name = "val"
        df2 = d_rate.reset_index()

        df2["age"] = "40-64"
        print(df2.head())

        return df2

    else:
        df = df[df.metric == "Rate"]
        return df


# 人口密度のデータフレーム、インプット
def d_mitsudo():
    with codecs.open("FEH_mitsudo.csv", "r", "Shift-JIS", "ignore") as file:
        d_mitsudo = pd.read_table(file, delimiter=",")

    d_mitsudo["year"] = d_mitsudo["time_code"] / 1000000
    d_mitsudo = d_mitsudo.rename(columns={'地域2010': 'kanji', 'value': 'population_density'})
    d_mitsudo["year"] = d_mitsudo['year'].astype('int')  # 人口密度のデータフレーム
    d_mitsudo = d_mitsudo[d_mitsudo.year.isin([2010])]
    d_mitsudo = d_mitsudo[["kanji", "population_density"]]

    # print(d_mitsudo.head())
    return d_mitsudo


# GBDからjapanmapを使ってマッピングするための関数
def GBD_japanmap_merge(df1):
    name_df = pd.DataFrame({"ID": list(range(1, 48)), "kanji": pref_names[1:48]})
    name_df['ID'].astype('int')  # IDと漢字の対応データフレーム

    pref_dict = {"Hokkaidō": "北海道",
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

    romaji = pref_dict.keys()
    kanji = pref_dict.values()
    rename_df = pd.DataFrame({"location": romaji, "kanji": kanji})

    # 漢字とローマ字の対応データフレーム
    kanji2romaji = pd.merge(name_df, rename_df)

    # 引数（GBDのデータフレーム）とマージさせる
    map_name = pd.merge(df1, kanji2romaji)
    return map_name  # japanmapのIDと、GBDのlocationを対応させたデータフレーム


# データフレームを作る部分

def japan_47P_incidence_df():
    # 47都道府県と漢字県名をマージ
    df1 = GBD_japanmap_merge(japan_47P_incidence())
    print("df1: ", df1)

    # 人口密度と発症率平均値のデータフレームをmerge
    df2 = pd.merge(d_mitsudo(), df1)
    print("df2: ", df2)

    csv_name = str(df2.age[0]).replace(' ', '')  # スペースを削除

    df2.to_csv("data47P_" + csv_name + ".csv")


# 解析用関数
def analysis_47P(df):  # dfには上で作成したデータフレームを代入

    title_age = df.age[0]  # 全年齢もしくは年齢別
    # 疾患を絞る★
    df = df[(df["cause"] == "Schizophrenia") | (df["cause"] == "Idiopathic epilepsy")]

    cause_groupby = df.groupby("cause")
    for name, group in cause_groupby:  # cause毎にデータフレームを分けて解析

        print("cause: ", name, "\n", group)

        # Part 1: 発症の平均値をマッピングする関数
        theme1 = "incidence (mean, male)"
        data1 = group[["location", "year", "val", "sex"]]
        print("data1: \n", data1)
        data1 = data1[data1.sex == "Male"]  # 男性か女性に絞る場合！

        data1 = data1.pivot_table(index="location", columns=["year", "sex"], values="val")
        d_mean = data1.mean(axis='columns')
        df_mean = pd.DataFrame(d_mean)  # 都道府県別の平均
        df_mean = df_mean.reset_index()
        df_mean = df_mean.rename(columns={0: "value_map"})
        print("df_mean: \n", df_mean)

        dg1 = pd.merge(group, df_mean)
        print("dg1: \n", dg1)

        # 都道府県マップに発症率平均値を描画
        mapping_colorscale(dg1, name, title_age, theme1)

        # 発症数のデータがあれば、人口密度との相関関係をプロット
        if not df_mean["value_map"].isnull().any():
            mapping_population_density(dg1, name, title_age, theme1)

        # Part 2: 発症の男女比の平均値をマッピング
        theme2 = "Female-Male ratio (mean)"

        d2 = group.set_index(["location", "year", "cause"])
        d2 = d2[["sex", "val"]]

        d2["ratio"] = d2[d2.sex == "Female"]["val"] / d2[d2.sex == "Male"]["val"]
        data = d2.pivot_table(index="location", columns=["year"], values="ratio")
        d_mean = data.mean(axis='columns')

        df_ratio_mean = pd.DataFrame(d_mean)  # 都道府県別の男女比の平均
        df_ratio_mean = df_ratio_mean.reset_index()
        df_ratio_mean = df_ratio_mean.rename(columns={0: "value_map"})

        dg2 = pd.merge(group, df_ratio_mean)
        print("dg2: \n", dg2)

        # 都道府県マップに発症率平均値を描画
        mapping_colorscale(dg2, name, title_age, theme2)

        # 発症数のデータがあれば、人口密度との相関関係をプロット
        if not df_ratio_mean["value_map"].isnull().any():
            mapping_population_density(dg2, name, title_age, theme2)

        # Part 3: 発症の減少率をマッピングする関数
        theme3 = "reduction rate of the male-female gap"

        dg4 = GBD_caliculator_kaiki(group)
        print("回帰係数の差のz score: \n", dg4)

        dg4 = pd.merge(group, dg4)

        mapping_colorscale(dg4, name, title_age, theme3)

        # 発症数のデータがあれば、人口密度との相関関係をプロット
        if not dg4["value_map"].isnull().any():
            mapping_population_density(dg4, name, title_age, theme3)

        if name == "Schizophrenia":
            return dg4

# GBDデータから男女比の減少率のデータフレームを計算する
def GBD_caliculator_kaiki(df2):
    df2 = df2[["ID", "year", "val", "sex"]]
    df2 = df2[df2.year.isin(YEARS)]  # 年を指定

    df2 = df2.pivot_table(index="ID", columns=["year", "sex"], values="val")

    # 回帰係数を計算
    model_lr = LinearRegression()
    m_coef = []
    f_coef = []

    for n in df2.index.values:  # 都道府県名毎に回帰係数を求める
        df = pd.DataFrame(df2.loc[n]).reset_index()

        male = df[df.sex == "Male"]
        male = male[["year", n]]
        m_x = male[["year"]]
        m_y = male[[n]]
        model_lr.fit(m_x, m_y)
        m_coef = m_coef + [[n, model_lr.coef_[0, 0]]]

        female = df[df.sex == "Female"]
        female = female[["year", n]]
        f_x = female[["year"]]
        f_y = female[[n]]
        model_lr.fit(f_x, f_y)
        f_coef = f_coef + [[n, model_lr.coef_[0, 0]]]

    df_m_coef = pd.DataFrame(m_coef, columns=['ID', 'm_coef'])
    df_f_coef = pd.DataFrame(f_coef, columns=['ID', 'f_coef'])
    df_coef = pd.merge(df_m_coef, df_f_coef)

    print(df_coef)

    df_coef["male_female"] = df_coef["m_coef"] - df_coef["f_coef"]  # 男性の減少率-女性の減少率
    mean = df_coef["male_female"].mean()
    std = df_coef["male_female"].std()
    # 標準化
    df_coef["value_map"] = (df_coef["male_female"] - mean) / std

    return df_coef


def mapping_colorscale(dg, name1, title_age1, theme):
    dg = dg[["ID", "value_map"]]
    dg = dg.set_index(["ID"])
    # 県ごとに色分けされて表示
    # テーマごとに色分け
    if theme == "reduction rate of the male-female gap":
        colors = 'Oranges_r'

    elif theme == "Female-Male ratio (mean)":
        colors = 'Blues_r'

    else:  # if theme == "incidence (mean)":
        colors = 'Reds'

    # japanmapのカラー
    # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
    # 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
    # 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
    # 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    # 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
    # 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    # 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',
    # 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r',
    # 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
    # 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r',
    # 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
    # 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2',
    # 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno',
    # 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r',
    # 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    # 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
    # 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r',
    # 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r',
    # 'winter', 'winter_r'

    # カラーマップと値を対応させるために、normで正規化
    cmap = plt.get_cmap(colors)
    norm = plt.Normalize(vmin=dg.value_map.min(), vmax=dg.value_map.max())

    # fcolは、人口の値を色に変換する関数
    fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()

    plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
    # pictureに、df1.人口.apply(fcol)を与えることで、人口ごとの色に塗る

    plt.title(name1 + "\n" + title_age1 + "\n" + theme)
    plt.imshow(picture(dg.value_map.apply(fcol)))
    plt.show()


def mapping_population_density(dg, name1, title_age1, theme):
    mitsudo = dg[["value_map", "population_density"]]
    mitsudo["population_density"] = mitsudo["population_density"].apply(np.log)  # 人口密度を対数変換
    print("人口密度と比較\n", mitsudo)

    print("人口密度との相関係数: ", mitsudo.corr().iloc[1, 0])

    s.jointplot(x=mitsudo["population_density"],
                y=mitsudo["value_map"],
                # xlim=[0,6500],
                color="k",
                data=mitsudo)  # 人口密度（横軸）の上限下限を

    # 相関係数を表示
    annotation1 = name1 + "\n" + title_age1 + "\n" + theme + "\n r=" + str(mitsudo.corr().iloc[1, 0])
    plt.title(annotation1, x=2, y=1.5)  # 相関係数を右上に表示
    plt.show()


# 全国データ
def japan_all_incidence():
    dfz = pd.read_csv("IHME-GBD_2019_DATA_JPN_15-39.csv", delimiter=",")
    print(dfz.head())

    title_age = dfz.age[0]
    # 40-64歳のデータを読み込んだ時のみ処理する
    if ((title_age != "All Ages") and (title_age != "15 to 39")) and (title_age != "65 to 89"):

        d = dfz.set_index(["year", "age", "sex", "cause"])
        print("1", d.head())

        d["population"] = d[d.metric == "Number"]["val"] / d[d.metric == "Rate"]["val"]  # 年齢別人口/10万人

        print("2", d.head())

        d = d.reset_index()
        print("3", d.head())

        d = d[d.metric == "Number"]  # 年齢別発症数

        # Age 40-44, ..., 60-64を合計して一つにまとめる

        d = d[["year", "age", "sex", "val", "population", "location", "cause"]]
        d_sum = d.groupby(["sex", "year", "cause"]).sum()  # 性別、年度毎に40-64の発症数と人口を合計
        d_rate = (d_sum["val"] / d_sum["population"])  # 性別、年齢ごとの発症率を計算
        d_rate.name = "val"
        df = d_rate.reset_index()

        df["age"] = "40-64"
        df["metric"] = "Rate"
        # df["cause"]=title
        # df.rename({'0':'val'},axis='columns')

        print("4", df.head())

        return df, title_age  # 40-64歳のデータフレーム

    else:
        dfz = dfz[dfz.metric == "Rate"]
        return dfz, title_age  # 40-64歳以外のデータフレーム


def japan_all_incidence_curve():
    df, title_age = japan_all_incidence()
    df = df[(df["cause"] == "Schizophrenia") | (df["cause"] == "Idiopathic epilepsy")]
    d = df[df.year.isin(YEARS)]  # 指定した年の中の変遷
    d = d.set_index(["cause", "year"])
    d = d[["sex", "val"]]
    ratio = d[d.sex == "Male"]["val"] / d[d.sex == "Female"]["val"]

    ax = s.lineplot(data=ratio.unstack(level=0))
    print("Male-Female incidence ratio, Age: ", title_age)
    # ax.set_ylim((0.8,1.4))
    # ax.set_xlim((1990,2020))
    plt.legend(bbox_to_anchor=(1.1, 1))  # 凡例は右上に
    # plt.title(title_z + "\n" + title_age + "\n Male Female ratio")
    plt.show()

    d = d.reset_index()

    cause_groupby = d.groupby("cause")
    for name, group in cause_groupby:  # cause毎にデータフレームを分けて解析

        # print("cause: ", name, "\n", group)
        cause_sex_groupby = group.groupby("sex")
        for sex, group_sex in cause_sex_groupby:
            x = group_sex["year"]  # 説明変数
            y = group_sex["val"]  # 目的変数

            # 全要素が1の列を説明変数の先頭に追加（絶対必要！！）
            X = sm.add_constant(x)

            # モデルの設定
            model = sm.OLS(y, X)

            # 回帰分析の実行
            results = model.fit()

            # 結果の詳細を表示
            print(name, sex, ", coef")
            print(results.summary())
            # SLR1 = sm.ols(formula="val ~ year", data=group_sex).fit()
            # print(SLR1.summary())
        colorlist = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        s.lmplot(x="year", y="val", hue="sex", markers=['o', 'X'], palette="binary",
                 data=group, legend_out=False)

        print(name, "\n Age: ", title_age)  # 年齢を表示
        title = name + title_age  # 年齢を表示
        plt.title(title)
        plt.show()
