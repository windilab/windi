import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold


# ============================================================
# 設定
# ============================================================

INPUT_FILE = Path("transition_shapley.csv")

# 出力ファイル
SHAPLEY_OUTPUT_FILE = Path("shapley_values_by_age.csv")
COALITION_OUTPUT_FILE = Path("estimated_coalition_values_by_age.csv")

FEATURES = ["body", "intel", "heart"]

AGE_LABELS = {
    0: "0-3",
    1: "4-6",
    2: "7-9",
    3: "10-12",
    4: "13-15",
    5: "16-18",
    6: "19-21",
    7: "22-24",
    8: "25-27",
    9: "28～30歳",
    10: "31～35歳",
    11: "36～40歳",
    12: "41～45歳",
    13: "46～50歳",
    14: "51歳以上",
}

# Ridge回帰で検討する正則化パラメータ
ALPHAS = np.logspace(-4, 4, 100)


# ============================================================
# データ読み込み・確認
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "age_ctgr",
    "body",
    "intel",
    "heart",
    "val",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"CSVに必要な列がありません: {missing_columns}"
    )

# 分析に使用する列を数値に変換
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# valが欠損している行は分析できないため除外
df = df.dropna(
    subset=["age_ctgr", "body", "intel", "heart", "val"]
).copy()

df["age_ctgr"] = df["age_ctgr"].astype(int)

# body、intel、heartが本当に0/1だけか確認
for feature in FEATURES:
    observed_values = set(df[feature].unique())

    if not observed_values.issubset({0, 1}):
        raise ValueError(
            f"{feature}に0/1以外の値があります: "
            f"{sorted(observed_values)}"
        )

    df[feature] = df[feature].astype(int)

print("読み込んだデータ")
print(df.head())

print("\n年齢区分別の回答数")
print(df.groupby("age_ctgr").size())


# ============================================================
# 8種類すべての組み合わせ
# ============================================================

all_combinations = pd.DataFrame(
    list(itertools.product([0, 1], repeat=3)),
    columns=FEATURES,
)

print("\n推定対象となる8種類の組み合わせ")
print(all_combinations)


# ============================================================
# 古典的シャープレイ値を計算する関数
# ============================================================

def calculate_shapley_values(value_dict, feature_names):
    """
    value_dict:
        キー   = frozenset形式の提携
        値     = 回帰モデルで推定された提携値

    例:
        frozenset()                  -> 何もなし
        frozenset({"body"})          -> bodyのみ
        frozenset({"body", "heart"}) -> body + heart
    """

    number_of_players = len(feature_names)
    shapley_values = {}

    for player in feature_names:
        shapley_value = 0.0

        other_players = [
            name for name in feature_names
            if name != player
        ]

        # 対象プレイヤーを含まないすべての提携を作る
        for coalition_size in range(len(other_players) + 1):
            for coalition_tuple in itertools.combinations(
                other_players,
                coalition_size
            ):
                coalition = frozenset(coalition_tuple)
                coalition_with_player = coalition | {player}

                weight = (
                    math.factorial(coalition_size)
                    * math.factorial(
                        number_of_players - coalition_size - 1
                    )
                    / math.factorial(number_of_players)
                )

                marginal_contribution = (
                    value_dict[coalition_with_player]
                    - value_dict[coalition]
                )

                shapley_value += weight * marginal_contribution

        shapley_values[player] = shapley_value

    return shapley_values


# ============================================================
# 年齢区分ごとの分析
# ============================================================

shapley_results = []
coalition_results = []

for age_ctgr in sorted(df["age_ctgr"].unique()):

    df_age = df[df["age_ctgr"] == age_ctgr].copy()

    age_label = AGE_LABELS.get(
        age_ctgr,
        f"不明な区分 {age_ctgr}"
    )

    print("\n" + "=" * 60)
    print(f"age_ctgr = {age_ctgr}: {age_label}")
    print(f"分析対象者数 = {len(df_age)}")

    # 回帰に必要な最低限のデータ数を確認
    if len(df_age) < 3:
        print("回答数が少なすぎるため、この年齢区分はスキップします。")
        continue

    X = df_age[FEATURES]
    y = df_age["val"]

    # --------------------------------------------------------
    # 各組み合わせの実際の回答者数と平均val
    # --------------------------------------------------------

    observed_summary = (
        df_age
        .groupby(FEATURES)["val"]
        .agg(
            observed_n="size",
            observed_mean_val="mean",
        )
        .reset_index()
    )

    combination_summary = all_combinations.merge(
        observed_summary,
        on=FEATURES,
        how="left",
    )

    combination_summary["observed_n"] = (
        combination_summary["observed_n"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Ridge回帰
    #
    # degree=3により以下をモデルに含める
    # body
    # intel
    # heart
    # body × intel
    # body × heart
    # intel × heart
    # body × intel × heart
    # --------------------------------------------------------

    number_of_folds = min(5, len(df_age))

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

    model.fit(X, y)

    # 8種類すべての組み合わせについてvalを推定
    predicted_values = model.predict(all_combinations)

    combination_summary["predicted_val"] = predicted_values
    combination_summary["age_ctgr"] = age_ctgr
    combination_summary["age_group"] = age_label

    # 観測された組み合わせかどうか
    combination_summary["combination_observed"] = np.where(
        combination_summary["observed_n"] > 0,
        "observed",
        "unobserved",
    )

    # 提携の分かりやすい名称
    def make_coalition_name(row):
        included = [
            feature
            for feature in FEATURES
            if row[feature] == 1
        ]

        if not included:
            return "none"

        return "+".join(included)

    combination_summary["coalition"] = combination_summary.apply(
        make_coalition_name,
        axis=1,
    )

    coalition_results.append(combination_summary)

    # --------------------------------------------------------
    # 推定した8種類のvalをシャープレイ値計算用の辞書に変換
    # --------------------------------------------------------

    value_dict = {}

    for _, row in combination_summary.iterrows():
        coalition = frozenset(
            feature
            for feature in FEATURES
            if row[feature] == 1
        )

        value_dict[coalition] = row["predicted_val"]

    # --------------------------------------------------------
    # シャープレイ値を計算
    # --------------------------------------------------------

    shapley_values = calculate_shapley_values(
        value_dict=value_dict,
        feature_names=FEATURES,
    )

    baseline_value = value_dict[frozenset()]
    full_value = value_dict[frozenset(FEATURES)]

    result = {
        "age_ctgr": age_ctgr,
        "age_group": age_label,
        "n": len(df_age),
        "observed_combinations":
            int((combination_summary["observed_n"] > 0).sum()),
        "missing_combinations":
            int((combination_summary["observed_n"] == 0).sum()),
        "baseline_predicted_val": baseline_value,
        "full_predicted_val": full_value,
        "body_shapley": shapley_values["body"],
        "intel_shapley": shapley_values["intel"],
        "heart_shapley": shapley_values["heart"],
        "shapley_sum": sum(shapley_values.values()),
        "full_minus_baseline": full_value - baseline_value,
    }

    shapley_results.append(result)

    print("\n組み合わせごとのval")
    print(
        combination_summary[
            [
                "coalition",
                "observed_n",
                "observed_mean_val",
                "predicted_val",
                "combination_observed",
            ]
        ].to_string(index=False)
    )

    print("\nシャープレイ値")
    print(f"body : {shapley_values['body']:.4f}")
    print(f"intel: {shapley_values['intel']:.4f}")
    print(f"heart: {shapley_values['heart']:.4f}")

    print(
        "シャープレイ値の合計: "
        f"{sum(shapley_values.values()):.4f}"
    )

    print(
        "全要因ありの予測値 − 全要因なしの予測値: "
        f"{full_value - baseline_value:.4f}"
    )


# ============================================================
# 結果の保存
# ============================================================

if not shapley_results:
    raise ValueError(
        "分析できる年齢区分がありませんでした。"
    )

shapley_df = pd.DataFrame(shapley_results)

coalition_df = pd.concat(
    coalition_results,
    ignore_index=True,
)

# 列の順番を整理
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

print("\n" + "=" * 60)
print("年齢区分別シャープレイ値")
print(shapley_df.to_string(index=False))

print("\n保存しました")
print(SHAPLEY_OUTPUT_FILE)
print(COALITION_OUTPUT_FILE)

# ============================================================
# シャープレイ値を各年齢区分で合計100に変換して可視化
# ============================================================

import matplotlib.pyplot as plt


# シャープレイ値の列
shapley_columns = [
    "body_shapley",
    "intel_shapley",
    "heart_shapley",
]

# 合計が0に近い場合は100への変換ができないため、NaNにする
near_zero = np.isclose(
    shapley_df["shapley_sum"],
    0,
    atol=1e-10,
)

if near_zero.any():
    skipped_age_groups = shapley_df.loc[
        near_zero,
        "age_group",
    ].tolist()

    print(
        "\nシャープレイ値の合計が0に近いため、"
        "割合を計算できない年齢区分があります:"
    )
    print(skipped_age_groups)


# 各年齢区分で合計100になるように変換
for column in shapley_columns:
    percentage_column = column.replace(
        "_shapley",
        "_shapley_percent",
    )

    shapley_df[percentage_column] = np.where(
        near_zero,
        np.nan,
        shapley_df[column]
        / shapley_df["shapley_sum"]
        * 100,
    )


# 変換後の合計を確認
shapley_df["shapley_percent_sum"] = (
    shapley_df[
        [
            "body_shapley_percent",
            "intel_shapley_percent",
            "heart_shapley_percent",
        ]
    ]
    .sum(axis=1, min_count=3)
)


print("\n各年齢区分で合計100に変換したシャープレイ値")
print(
    shapley_df[
        [
            "age_ctgr",
            "age_group",
            "body_shapley_percent",
            "intel_shapley_percent",
            "heart_shapley_percent",
            "shapley_percent_sum",
        ]
    ].to_string(index=False)
)


# ============================================================
# 変換後の結果をCSVに保存
# ============================================================

SHAPLEY_PERCENT_OUTPUT_FILE = Path(
    "shapley_values_by_age_percent.csv"
)

shapley_df.to_csv(
    SHAPLEY_PERCENT_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n保存しました")
print(SHAPLEY_PERCENT_OUTPUT_FILE)


# ============================================================
# 折れ線グラフ
# ============================================================

# 年齢区分の順番を明示的に並べる
plot_df = (
    shapley_df
    .sort_values("age_ctgr")
    .reset_index(drop=True)
)

x = np.arange(len(plot_df))

plt.figure(figsize=(13, 7))

plt.plot(
    x,
    plot_df["body_shapley_percent"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="#0072B2",
    label="Physical",
)

plt.plot(
    x,
    plot_df["intel_shapley_percent"],
    marker="s",
    linewidth=2,
    markersize=6,
    color="#E69F00",
    label="Intellectual",
)

plt.plot(
    x,
    plot_df["heart_shapley_percent"],
    marker="^",
    linewidth=2,
    markersize=7,
    color="#CC79A7",
    label="Mental",
)

# シャープレイ値が負になる場合も分かるように0の基準線を表示
plt.axhline(
    y=0,
    color="black",
    linewidth=0.8,
    linestyle="--",
)

plt.xticks(
    ticks=x,
    labels=plot_df["age_group"],
    rotation=45,
    ha="right",
)

plt.xlabel("Age group", fontsize=12)
plt.ylabel("Shapley value (%)", fontsize=12)
plt.title(
    "Relative Shapley values by age group",
    fontsize=14,
)

plt.legend(
    title="Domain",
    frameon=False,
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

plt.tight_layout()

# 画像として保存
SHAPLEY_PLOT_FILE = Path(
    "shapley_values_by_age_percent.png"
)

plt.savefig(
    SHAPLEY_PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nグラフを保存しました")
print(SHAPLEY_PLOT_FILE)

# ============================================================
# 100に変換していない元のシャープレイ値を可視化
# ============================================================

# 年齢区分順に並べる
plot_df_raw = (
    shapley_df
    .sort_values("age_ctgr")
    .reset_index(drop=True)
)

x = np.arange(len(plot_df_raw))

plt.figure(figsize=(13, 7))

plt.plot(
    x,
    plot_df_raw["body_shapley"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="#0072B2",
    label="Physical",
)

plt.plot(
    x,
    plot_df_raw["intel_shapley"],
    marker="s",
    linewidth=2,
    markersize=6,
    color="#E69F00",
    label="Intellectual",
)

plt.plot(
    x,
    plot_df_raw["heart_shapley"],
    marker="^",
    linewidth=2,
    markersize=7,
    color="#CC79A7",
    label="Mental",
)

# 正負を確認しやすくするための基準線
plt.axhline(
    y=0,
    color="black",
    linewidth=0.8,
    linestyle="--",
)

plt.xticks(
    ticks=x,
    labels=plot_df_raw["age_group"],
    rotation=45,
    ha="right",
)

plt.xlabel("Age group", fontsize=12)
plt.ylabel("Shapley value", fontsize=12)
plt.title(
    "Raw Shapley values by age group",
    fontsize=14,
)

plt.legend(
    title="Domain",
    frameon=False,
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

plt.tight_layout()

# 画像として保存
SHAPLEY_RAW_PLOT_FILE = Path(
    "shapley_values_by_age_raw.png"
)

plt.savefig(
    SHAPLEY_RAW_PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\n100に変換していないグラフを保存しました")
print(SHAPLEY_RAW_PLOT_FILE)
