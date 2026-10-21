import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================
# 設定
# ============================================================

INPUT_FILE = Path("transition_shapley.csv")

# 出力ファイル
SHAPLEY_OUTPUT_FILE = Path(
    "shapley_values_by_age.csv"
)

COALITION_OUTPUT_FILE = Path(
    "estimated_coalition_values_by_age.csv"
)

SHAPLEY_PLOT_FILE = Path(
    "shapley_values_by_age_raw.png"
)

FEATURES = [
    "body",
    "intel",
    "heart",
]


# ============================================================
# 年齢区分ラベル
# ============================================================

AGE_LABELS = {
    0: "0～3歳",
    1: "4～6歳",
    2: "7～9歳",
    3: "10～12歳",
    4: "13～15歳",
    5: "16～18歳",
    6: "19～21歳",
    7: "22～24歳",
    8: "25～27歳",
    9: "28～30歳",
    10: "31～35歳",
    11: "36～40歳",
    12: "41～45歳",
    13: "46～50歳",
    14: "51歳以上",
}


# Ridge回帰で検討する正則化パラメータ
ALPHAS = np.logspace(
    -4,
    4,
    100,
)


# ============================================================
# データ読み込み・確認
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "age_ctgr",
    "body",
    "intel",
    "heart",
    "val",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:
    raise ValueError(
        f"CSVに必要な列がありません: "
        f"{missing_columns}"
    )


# ------------------------------------------------------------
# 分析に使用する列を数値に変換
# ------------------------------------------------------------

for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ------------------------------------------------------------
# 欠損値を除外
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        "age_ctgr",
        "body",
        "intel",
        "heart",
        "val",
    ]
).copy()


df["age_ctgr"] = (
    df["age_ctgr"]
    .astype(int)
)


# ------------------------------------------------------------
# body, intel, heartが0/1か確認
# ------------------------------------------------------------

for feature in FEATURES:

    observed_values = set(
        df[feature].unique()
    )

    if not observed_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"{feature}に0/1以外の値があります: "
            f"{sorted(observed_values)}"
        )

    df[feature] = (
        df[feature]
        .astype(int)
    )


print(
    "読み込んだデータ"
)

print(
    df.head()
)


print(
    "\n年齢区分別の回答数"
)

print(
    df
    .groupby("age_ctgr")
    .size()
)


# ============================================================
# 8種類すべての組み合わせ
# ============================================================

all_combinations = pd.DataFrame(
    list(
        itertools.product(
            [0, 1],
            repeat=3,
        )
    ),
    columns=FEATURES,
)


print(
    "\n推定対象となる8種類の組み合わせ"
)

print(
    all_combinations
)


# ============================================================
# 古典的シャープレイ値を計算する関数
# ============================================================

def calculate_shapley_values(
    value_dict,
    feature_names,
):
    """
    value_dict:
        キー = frozenset形式の提携
        値   = 回帰モデルで推定された提携値

    例:
        frozenset()
            -> 何もなし

        frozenset({"body"})
            -> bodyのみ

        frozenset({"body", "heart"})
            -> body + heart
    """

    number_of_players = len(
        feature_names
    )

    shapley_values = {}


    for player in feature_names:

        shapley_value = 0.0


        other_players = [
            name
            for name in feature_names
            if name != player
        ]


        # ----------------------------------------------------
        # 対象プレイヤーを含まないすべての提携を作る
        # ----------------------------------------------------

        for coalition_size in range(
            len(other_players) + 1
        ):

            for coalition_tuple in itertools.combinations(
                other_players,
                coalition_size,
            ):

                coalition = frozenset(
                    coalition_tuple
                )

                coalition_with_player = (
                    coalition | {player}
                )


                weight = (
                    math.factorial(
                        coalition_size
                    )
                    *
                    math.factorial(
                        number_of_players
                        - coalition_size
                        - 1
                    )
                    /
                    math.factorial(
                        number_of_players
                    )
                )


                marginal_contribution = (
                    value_dict[
                        coalition_with_player
                    ]
                    -
                    value_dict[
                        coalition
                    ]
                )


                shapley_value += (
                    weight
                    *
                    marginal_contribution
                )


        shapley_values[
            player
        ] = shapley_value


    return shapley_values


# ============================================================
# 年齢区分ごとの分析
# ============================================================

shapley_results = []

coalition_results = []


for age_ctgr in sorted(
    df["age_ctgr"].unique()
):


    df_age = df[
        df["age_ctgr"] == age_ctgr
    ].copy()


    age_label = AGE_LABELS.get(
        age_ctgr,
        f"不明な区分 {age_ctgr}",
    )


    print(
        "\n" + "=" * 60
    )

    print(
        f"age_ctgr = {age_ctgr}: "
        f"{age_label}"
    )

    print(
        f"分析対象者数 = "
        f"{len(df_age)}"
    )


    # --------------------------------------------------------
    # 回帰に必要な最低限のデータ数を確認
    # --------------------------------------------------------

    if len(df_age) < 3:

        print(
            "回答数が少なすぎるため、"
            "この年齢区分はスキップします。"
        )

        continue


    X = df_age[
        FEATURES
    ]

    y = df_age[
        "val"
    ]


    # ========================================================
    # 各組み合わせの実際の回答者数と平均val
    # ========================================================

    observed_summary = (
        df_age
        .groupby(
            FEATURES
        )["val"]
        .agg(
            observed_n="size",
            observed_mean_val="mean",
        )
        .reset_index()
    )


    combination_summary = (
        all_combinations
        .merge(
            observed_summary,
            on=FEATURES,
            how="left",
        )
    )


    combination_summary[
        "observed_n"
    ] = (
        combination_summary[
            "observed_n"
        ]
        .fillna(0)
        .astype(int)
    )


    # ========================================================
    # Ridge回帰
    #
    # degree=3により以下を含める
    #
    # body
    # intel
    # heart
    # body × intel
    # body × heart
    # intel × heart
    # body × intel × heart
    # ========================================================

    number_of_folds = min(
        5,
        len(df_age),
    )


    cv = KFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=42,
    )


    model = make_pipeline(

        PolynomialFeatures(
            degree=3,
            interaction_only=True,
            include_bias=False,
        ),

        RidgeCV(
            alphas=ALPHAS,
            cv=cv,
        ),
    )


    model.fit(
        X,
        y,
    )


    # --------------------------------------------------------
    # 8種類すべてについてvalを推定
    # --------------------------------------------------------

    predicted_values = (
        model.predict(
            all_combinations
        )
    )


    combination_summary[
        "predicted_val"
    ] = predicted_values


    combination_summary[
        "age_ctgr"
    ] = age_ctgr


    combination_summary[
        "age_group"
    ] = age_label


    # --------------------------------------------------------
    # 観測された組み合わせかどうか
    # --------------------------------------------------------

    combination_summary[
        "combination_observed"
    ] = np.where(

        combination_summary[
            "observed_n"
        ] > 0,

        "observed",

        "unobserved",
    )


    # ========================================================
    # 提携名称
    # ========================================================

    def make_coalition_name(row):

        included = [
            feature
            for feature in FEATURES
            if row[feature] == 1
        ]

        if not included:
            return "none"

        return "+".join(
            included
        )


    combination_summary[
        "coalition"
    ] = combination_summary.apply(
        make_coalition_name,
        axis=1,
    )


    coalition_results.append(
        combination_summary
    )


    # ========================================================
    # シャープレイ値計算用の辞書
    # ========================================================

    value_dict = {}


    for _, row in combination_summary.iterrows():

        coalition = frozenset(
            feature
            for feature in FEATURES
            if row[feature] == 1
        )

        value_dict[
            coalition
        ] = row[
            "predicted_val"
        ]


    # ========================================================
    # シャープレイ値を計算
    # ========================================================

    shapley_values = (
        calculate_shapley_values(
            value_dict=value_dict,
            feature_names=FEATURES,
        )
    )


    baseline_value = (
        value_dict[
            frozenset()
        ]
    )


    full_value = (
        value_dict[
            frozenset(
                FEATURES
            )
        ]
    )


    result = {

        "age_ctgr":
            age_ctgr,

        "age_group":
            age_label,

        "n":
            len(df_age),

        "observed_combinations":
            int(
                (
                    combination_summary[
                        "observed_n"
                    ] > 0
                ).sum()
            ),

        "missing_combinations":
            int(
                (
                    combination_summary[
                        "observed_n"
                    ] == 0
                ).sum()
            ),

        "baseline_predicted_val":
            baseline_value,

        "full_predicted_val":
            full_value,

        "body_shapley":
            shapley_values[
                "body"
            ],

        "intel_shapley":
            shapley_values[
                "intel"
            ],

        "heart_shapley":
            shapley_values[
                "heart"
            ],

        "shapley_sum":
            sum(
                shapley_values.values()
            ),

        "full_minus_baseline":
            full_value
            -
            baseline_value,
    }


    shapley_results.append(
        result
    )


    # ========================================================
    # 結果表示
    # ========================================================

    print(
        "\n組み合わせごとのval"
    )


    print(
        combination_summary[
            [
                "coalition",
                "observed_n",
                "observed_mean_val",
                "predicted_val",
                "combination_observed",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\n困難度"
    )


    print(
        f"body : "
        f"{shapley_values['body']:.4f}"
    )

    print(
        f"intel: "
        f"{shapley_values['intel']:.4f}"
    )

    print(
        f"heart: "
        f"{shapley_values['heart']:.4f}"
    )


    print(
        "シャープレイ値の合計: "
        f"{sum(shapley_values.values()):.4f}"
    )


    print(
        "全要因ありの予測値 − "
        "全要因なしの予測値: "
        f"{full_value - baseline_value:.4f}"
    )


# ============================================================
# 結果の保存
# ============================================================

if not shapley_results:

    raise ValueError(
        "分析できる年齢区分がありませんでした。"
    )


shapley_df = pd.DataFrame(
    shapley_results
)


coalition_df = pd.concat(
    coalition_results,
    ignore_index=True,
)


# ============================================================
# 列の順番を整理
# ============================================================

shapley_df = shapley_df[
    [
        "age_ctgr",
        "age_group",
        "n",
        "observed_combinations",
        "missing_combinations",
        "baseline_predicted_val",
        "full_predicted_val",
        "body_shapley",
        "intel_shapley",
        "heart_shapley",
        "shapley_sum",
        "full_minus_baseline",
    ]
]


coalition_df = coalition_df[
    [
        "age_ctgr",
        "age_group",
        "coalition",
        "body",
        "intel",
        "heart",
        "observed_n",
        "observed_mean_val",
        "predicted_val",
        "combination_observed",
    ]
]


shapley_df.to_csv(
    SHAPLEY_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


coalition_df.to_csv(
    COALITION_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\n" + "=" * 60
)

print(
    "年齢区分別シャープレイ値"
)

print(
    shapley_df.to_string(
        index=False
    )
)


print(
    "\n保存しました"
)

print(
    SHAPLEY_OUTPUT_FILE
)

print(
    COALITION_OUTPUT_FILE
)


# ============================================================
# Figure用 日本語フォント設定
# ============================================================

japanese_font_candidates = [

    "Yu Gothic",

    "YuGothic",

    "Meiryo",

    "Noto Sans CJK JP",

    "IPAexGothic",

    "IPAGothic",
]


available_fonts = {
    font.name
    for font in fm.fontManager.ttflist
}


for font_name in japanese_font_candidates:

    if font_name in available_fonts:

        plt.rcParams[
            "font.family"
        ] = font_name

        print(
            f"\n日本語フォントとして "
            f"{font_name} を使用します。"
        )

        break

else:

    print(
        "\n日本語フォントが見つかりませんでした。"
    )

    print(
        "日本語が文字化けする場合は、"
        "日本語フォントをインストールしてください。"
    )


# ------------------------------------------------------------
# matplotlib全体のフォント設定
# ------------------------------------------------------------

plt.rcParams[
    "axes.unicode_minus"
] = False


plt.rcParams[
    "font.size"
] = 14


plt.rcParams[
    "axes.labelsize"
] = 17


plt.rcParams[
    "xtick.labelsize"
] = 14


plt.rcParams[
    "ytick.labelsize"
] = 14


plt.rcParams[
    "legend.fontsize"
] = 15


# ============================================================
# シャープレイ値のFigure
# ============================================================

plot_df = (
    shapley_df
    .sort_values(
        "age_ctgr"
    )
    .reset_index(
        drop=True
    )
)


x = np.arange(
    len(plot_df)
)


# ------------------------------------------------------------
# Figure作成
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(15, 8)
)


# ============================================================
# 身体領域
#
# ○ 実線
# ============================================================

ax.plot(

    x,

    plot_df[
        "body_shapley"
    ],

    marker="o",

    linestyle="-",

    linewidth=2.8,

    markersize=9,

    color="black",

    markerfacecolor="white",

    markeredgecolor="black",

    markeredgewidth=1.8,

    label="身体領域",
)


# ============================================================
# 知的領域
#
# □ 破線
# ============================================================

ax.plot(

    x,

    plot_df[
        "intel_shapley"
    ],

    marker="s",

    linestyle="--",

    linewidth=2.8,

    markersize=9,

    color="black",

    markerfacecolor="white",

    markeredgecolor="black",

    markeredgewidth=1.8,

    label="知的領域",
)


# ============================================================
# 精神領域
#
# △ 点線
# ============================================================

ax.plot(

    x,

    plot_df[
        "heart_shapley"
    ],

    marker="^",

    linestyle=":",

    linewidth=3.0,

    markersize=10,

    color="black",

    markerfacecolor="white",

    markeredgecolor="black",

    markeredgewidth=1.8,

    label="精神領域",
)


# ============================================================
# 0の基準線
# ============================================================

ax.axhline(

    y=0,

    color="black",

    linewidth=1.0,

    linestyle="-",

    alpha=0.5,

    zorder=0,
)


# ============================================================
# X軸
# ============================================================

ax.set_xticks(
    x
)


ax.set_xticklabels(

    plot_df[
        "age_group"
    ],

    rotation=0,

    ha="right",

    fontsize=14,
)


# ============================================================
# X軸ラベル
# ============================================================

ax.set_xlabel(

    "年齢区分",

    fontsize=22,

    fontweight="bold",

    labelpad=14,
)


# ============================================================
# Y軸ラベル
# ============================================================

ax.set_ylabel(

    "困難度",

    fontsize=32,

    fontweight="bold",

    labelpad=14,
)


# ============================================================
# 凡例
# ============================================================

ax.legend(

    title="領域",

    title_fontsize=22,

    fontsize=22,

    frameon=False,

    loc="best",

    handlelength=3.5,

    handletextpad=0.8,
)


# ============================================================
# Y軸方向のみ補助線
# ============================================================

ax.grid(

    axis="y",

    linestyle="--",

    linewidth=0.8,

    alpha=0.25,

    zorder=0,
)


# X軸方向のグリッドは表示しない
ax.grid(

    axis="x",

    visible=False,
)


# ============================================================
# 外枠
# ============================================================

ax.spines[
    "top"
].set_visible(
    False
)


ax.spines[
    "right"
].set_visible(
    False
)


ax.spines[
    "left"
].set_linewidth(
    1.3
)


ax.spines[
    "bottom"
].set_linewidth(
    1.3
)


# ============================================================
# 軸目盛り
# ============================================================

ax.tick_params(

    axis="both",

    width=1.2,

    length=5,

    labelsize=22,
)


# ============================================================
# レイアウト調整
# ============================================================

plt.tight_layout()


# ============================================================
# 高解像度で保存
# ============================================================

plt.savefig(

    SHAPLEY_PLOT_FILE,

    dpi=600,

    bbox_inches="tight",
)


plt.show()


print(
    "\nシャープレイ値のグラフを保存しました"
)

print(
    SHAPLEY_PLOT_FILE
)
