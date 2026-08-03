"""
都道府県別・男女別・年次別パネルデータを用いた固定効果分析

主解析：
    log(発症率)
    = 所得
    + 所得 × 女性
    + 都道府県 × 性別固定効果
    + 性別 × 年固定効果

標準誤差：
    都道府県単位のクラスターロバスト標準誤差

必要なパッケージ：
    pip install pandas numpy scipy linearmodels
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from linearmodels.iv.absorbing import AbsorbingLS


# ============================================================
# 1. ユーザー設定
# ============================================================

FILE_PATH = Path("data.csv")

# 実際のデータの列名に合わせて変更してください
PREF_COL = "prefecture"
SEX_COL = "sex"
YEAR_COL = "year"
RATE_COL = "incidence_rate"
INCOME_COL = "income"

# 性別コードを女性=1、男性=0に変換します
# データに合わせて変更してください
SEX_MAP = {
    "Male": 0,
    "Female": 1,
    "男性": 0,
    "女性": 1,

    # 数値コードの場合の例
    # 1: 0,  # 男性
    # 2: 1,  # 女性
}

# 所得の解析単位
#
# 所得が「円」の場合：
# 1_000_000で割ると、所得100万円増加当たりの関連になります。
#
# 所得が「万円」の場合：
# 100で割ると、所得100万円増加当たりの関連になります。
#
# すでに100万円単位なら1にしてください。
INCOME_DIVISOR = 1_000_000

# 都道府県・性別・年ごとに変動する共変量があれば追加します
TIME_VARYING_COVARIATES = [
    # "unemployment_rate",
    # "population_density",
]

OUTPUT_DIR = Path("panel_model_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. データの読み込みと前処理
# ============================================================

def prepare_data(file_path: Path) -> pd.DataFrame:
    """データを読み込み、解析用変数を作成する。"""

    df = pd.read_csv(file_path)

    required_cols = [
        PREF_COL,
        SEX_COL,
        YEAR_COL,
        RATE_COL,
        INCOME_COL,
        *TIME_VARYING_COVARIATES,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            "次の列がデータにありません："
            + ", ".join(missing_cols)
        )

    df = df[required_cols].copy()

    # 数値変数を数値型に変換
    df[YEAR_COL] = pd.to_numeric(
        df[YEAR_COL],
        errors="coerce",
    )

    df[RATE_COL] = pd.to_numeric(
        df[RATE_COL],
        errors="coerce",
    )

    df[INCOME_COL] = pd.to_numeric(
        df[INCOME_COL],
        errors="coerce",
    )

    for col in TIME_VARYING_COVARIATES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # 女性=1、男性=0
    df["female"] = df[SEX_COL].map(SEX_MAP)

    unknown_sex = df.loc[
        df["female"].isna() & df[SEX_COL].notna(),
        SEX_COL,
    ].unique()

    if len(unknown_sex) > 0:
        raise ValueError(
            "SEX_MAPに登録されていない性別コードがあります："
            f"{unknown_sex.tolist()}\n"
            "SEX_MAPをデータに合わせて修正してください。"
        )

    # 解析に必要な変数の欠損を除外
    before_n = len(df)

    df = df.dropna(
        subset=[
            PREF_COL,
            SEX_COL,
            YEAR_COL,
            RATE_COL,
            INCOME_COL,
            "female",
            *TIME_VARYING_COVARIATES,
        ]
    ).copy()

    after_n = len(df)

    print("欠損除外")
    print(f"  除外前：{before_n:,}行")
    print(f"  除外後：{after_n:,}行")
    print(f"  除外数：{before_n - after_n:,}行")

    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df["female"] = df["female"].astype(int)

    # 都道府県×性別×年の重複を確認
    key_cols = [
        PREF_COL,
        SEX_COL,
        YEAR_COL,
    ]

    duplicated = df.duplicated(
        key_cols,
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = (
            df.loc[duplicated, key_cols]
            .sort_values(key_cols)
            .head(20)
        )

        raise ValueError(
            "都道府県×性別×年が重複しています。\n"
            "各組み合わせを1行にしてください。\n\n"
            f"{duplicate_rows}"
        )

    # 発症率が対数変換できるか確認
    nonpositive = df[RATE_COL] <= 0

    if nonpositive.any():
        raise ValueError(
            f"発症率が0以下の行が"
            f"{int(nonpositive.sum())}行あります。\n"
            "対数変換できないため、値を確認してください。"
        )

    # 所得の単位を変更
    df["income_scaled"] = (
        df[INCOME_COL] / INCOME_DIVISOR
    )

    # 発症率の自然対数
    df["log_incidence_rate"] = np.log(
        df[RATE_COL]
    )

    # 都道府県×性別固定効果
    df["prefecture_sex_fe"] = (
        df[PREF_COL].astype(str)
        + "__"
        + df["female"].astype(str)
    ).astype("category")

    # 性別×年固定効果
    df["sex_year_fe"] = (
        df["female"].astype(str)
        + "__"
        + df[YEAR_COL].astype(str)
    ).astype("category")

    # 都道府県×性別内の所得平均との差
    # 記述・確認用です
    df["income_within"] = (
        df["income_scaled"]
        - df.groupby(
            [PREF_COL, "female"]
        )["income_scaled"].transform("mean")
    )

    df = df.sort_values(
        [PREF_COL, "female", YEAR_COL]
    ).reset_index(drop=True)

    return df


# ============================================================
# 3. 固定効果モデル
# ============================================================

def fit_fixed_effects_model(
    data: pd.DataFrame,
    outcome_col: str,
    income_col: str,
    covariates: Optional[list[str]] = None,
):
    """
    固定効果モデルを推定する。

    outcome
    = 所得
    + 所得×女性
    + 共変量
    + 都道府県×性別固定効果
    + 性別×年固定効果
    + 誤差
    """

    if covariates is None:
        covariates = []

    df_model = data.copy()

    interaction_col = (
        f"{income_col}_x_female"
    )

    df_model[interaction_col] = (
        df_model[income_col]
        * df_model["female"]
    )

    explanatory_cols = [
        income_col,
        interaction_col,
        *covariates,
    ]

    required_cols = [
        outcome_col,
        PREF_COL,
        "prefecture_sex_fe",
        "sex_year_fe",
        *explanatory_cols,
    ]

    df_model = df_model.dropna(
        subset=required_cols
    ).copy()

    if len(df_model) == 0:
        raise ValueError(
            "解析可能な観測がありません。"
        )

    # 関心のある説明変数
    exog = df_model[
        explanatory_cols
    ].astype(float)

    # 吸収する固定効果
    absorb = pd.DataFrame(
        {
            "prefecture_sex_fe":
                pd.Categorical(
                    df_model["prefecture_sex_fe"]
                ),
            "sex_year_fe":
                pd.Categorical(
                    df_model["sex_year_fe"]
                ),
        },
        index=df_model.index,
    )

    model = AbsorbingLS(
        dependent=df_model[
            outcome_col
        ].astype(float),
        exog=exog,
        absorb=absorb,
        drop_absorbed=True,
    )

    # 都道府県単位のクラスタリング
    cluster_id = pd.Categorical(
        df_model[PREF_COL]
    ).codes

    result = model.fit(
        cov_type="clustered",
        clusters=cluster_id,
        debiased=True,
    )

    return (
        result,
        df_model,
        income_col,
        interaction_col,
    )


# ============================================================
# 4. 係数の線形結合
# ============================================================

def calculate_linear_combination(
    result,
    weights: dict[str, float],
    n_clusters: int,
    label: str,
    log_outcome: bool,
) -> dict:
    """
    男性係数、女性係数、男女差などの線形結合を計算する。
    """

    parameter_names = list(
        result.params.index
    )

    weight_vector = pd.Series(
        0.0,
        index=parameter_names,
        dtype=float,
    )

    for parameter, weight in weights.items():
        if parameter not in weight_vector.index:
            raise KeyError(
                f"{parameter}がモデル内にありません。"
            )

        weight_vector.loc[parameter] = weight

    coefficient = float(
        weight_vector @ result.params
    )

    covariance = result.cov.loc[
        parameter_names,
        parameter_names,
    ]

    variance = float(
        weight_vector.to_numpy()
        @ covariance.to_numpy()
        @ weight_vector.to_numpy()
    )

    variance = max(variance, 0.0)
    standard_error = np.sqrt(variance)

    # 都道府県クラスター数−1のt分布
    degrees_freedom = n_clusters - 1

    critical_value = student_t.ppf(
        0.975,
        df=degrees_freedom,
    )

    ci_lower = (
        coefficient
        - critical_value * standard_error
    )

    ci_upper = (
        coefficient
        + critical_value * standard_error
    )

    if standard_error > 0:
        test_statistic = (
            coefficient / standard_error
        )

        p_value = 2 * student_t.sf(
            abs(test_statistic),
            df=degrees_freedom,
        )
    else:
        test_statistic = np.nan
        p_value = np.nan

    output = {
        "comparison": label,
        "coefficient": coefficient,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "test_statistic": test_statistic,
        "p_value": p_value,
    }

    # 対数アウトカムの場合は百分率変化も計算
    if log_outcome:
        output["percent_change"] = (
            100 * np.expm1(coefficient)
        )

        output["percent_ci_lower"] = (
            100 * np.expm1(ci_lower)
        )

        output["percent_ci_upper"] = (
            100 * np.expm1(ci_upper)
        )

    return output


def summarize_sex_specific_effects(
    result,
    data: pd.DataFrame,
    income_col: str,
    interaction_col: str,
    log_outcome: bool,
) -> pd.DataFrame:
    """
    男性、女性、男女差の結果を表形式で返す。
    """

    n_clusters = data[PREF_COL].nunique()

    results = [
        # 男性：
        # beta_income
        calculate_linear_combination(
            result=result,
            weights={
                income_col: 1,
                interaction_col: 0,
            },
            n_clusters=n_clusters,
            label="男性における所得の関連",
            log_outcome=log_outcome,
        ),

        # 女性：
        # beta_income + beta_interaction
        calculate_linear_combination(
            result=result,
            weights={
                income_col: 1,
                interaction_col: 1,
            },
            n_clusters=n_clusters,
            label="女性における所得の関連",
            log_outcome=log_outcome,
        ),

        # 男女差：
        # beta_interaction
        calculate_linear_combination(
            result=result,
            weights={
                income_col: 0,
                interaction_col: 1,
            },
            n_clusters=n_clusters,
            label="所得との関連の男女差（女性－男性）",
            log_outcome=log_outcome,
        ),
    ]

    return pd.DataFrame(results)


# ============================================================
# 5. 1年ラグ所得の作成
# ============================================================

def add_one_year_lag(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    ある年の発症率に厳密に1年前の所得を対応させる。

    例：
        2001年の発症率
        ← 2000年の所得
    """

    lag_data = data[
        [
            PREF_COL,
            "female",
            YEAR_COL,
            "income_scaled",
        ]
    ].copy()

    # 元の所得を翌年のアウトカムに対応
    lag_data[YEAR_COL] = (
        lag_data[YEAR_COL] + 1
    )

    lag_data = lag_data.rename(
        columns={
            "income_scaled": "income_lag1"
        }
    )

    merged = data.merge(
        lag_data,
        on=[
            PREF_COL,
            "female",
            YEAR_COL,
        ],
        how="left",
        validate="one_to_one",
    )

    return merged


# ============================================================
# 6. 解析の実行
# ============================================================

def main() -> None:
    df = prepare_data(FILE_PATH)

    print()
    print("=" * 60)
    print("データ概要")
    print("=" * 60)

    print(f"観測数：{len(df):,}")
    print(
        f"都道府県数："
        f"{df[PREF_COL].nunique()}"
    )
    print(
        f"対象年数："
        f"{df[YEAR_COL].nunique()}"
    )
    print(
        f"対象期間："
        f"{df[YEAR_COL].min()}"
        f"～"
        f"{df[YEAR_COL].max()}"
    )
    print(
        f"所得の全体SD："
        f"{df['income_scaled'].std():.4f}"
    )
    print(
        f"所得のwithin SD："
        f"{df['income_within'].std():.4f}"
    )

    if np.isclose(
        df["income_within"].std(),
        0,
    ):
        raise ValueError(
            "同じ都道府県・性別内で"
            "所得がほとんど変化していません。\n"
            "この場合、固定効果モデルでは"
            "所得変化の関連を推定できません。"
        )

    # ========================================================
    # 主解析：対数発症率
    # ========================================================

    (
        main_result,
        main_data,
        main_income,
        main_interaction,
    ) = fit_fixed_effects_model(
        data=df,
        outcome_col="log_incidence_rate",
        income_col="income_scaled",
        covariates=TIME_VARYING_COVARIATES,
    )

    print()
    print("=" * 60)
    print("主解析：対数発症率")
    print("=" * 60)
    print(main_result.summary)

    main_effects = (
        summarize_sex_specific_effects(
            result=main_result,
            data=main_data,
            income_col=main_income,
            interaction_col=main_interaction,
            log_outcome=True,
        )
    )

    print()
    print("男女別の所得との関連")
    print(
        main_effects
        .round(4)
        .to_string(index=False)
    )

    main_effects.to_csv(
        OUTPUT_DIR
        / "main_model_sex_specific_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        OUTPUT_DIR / "main_model_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(main_result.summary))

    # ========================================================
    # 感度分析1：非対数発症率
    # ========================================================

    (
        raw_result,
        raw_data,
        raw_income,
        raw_interaction,
    ) = fit_fixed_effects_model(
        data=df,
        outcome_col=RATE_COL,
        income_col="income_scaled",
        covariates=TIME_VARYING_COVARIATES,
    )

    raw_effects = (
        summarize_sex_specific_effects(
            result=raw_result,
            data=raw_data,
            income_col=raw_income,
            interaction_col=raw_interaction,
            log_outcome=False,
        )
    )

    print()
    print("=" * 60)
    print("感度分析1：非対数発症率")
    print("=" * 60)

    print(
        raw_effects
        .round(4)
        .to_string(index=False)
    )

    raw_effects.to_csv(
        OUTPUT_DIR
        / "sensitivity_raw_incidence_rate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 感度分析2：1年前の所得
    # ========================================================

    df_lag = add_one_year_lag(df)

    (
        lag_result,
        lag_data,
        lag_income,
        lag_interaction,
    ) = fit_fixed_effects_model(
        data=df_lag,
        outcome_col="log_incidence_rate",
        income_col="income_lag1",
        covariates=TIME_VARYING_COVARIATES,
    )

    lag_effects = (
        summarize_sex_specific_effects(
            result=lag_result,
            data=lag_data,
            income_col=lag_income,
            interaction_col=lag_interaction,
            log_outcome=True,
        )
    )

    print()
    print("=" * 60)
    print("感度分析2：1年前の所得")
    print("=" * 60)

    print(
        lag_effects
        .round(4)
        .to_string(index=False)
    )

    lag_effects.to_csv(
        OUTPUT_DIR
        / "sensitivity_income_lag1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 全結果を結合して保存
    # ========================================================

    main_effects["model"] = (
        "Main: log incidence rate"
    )

    raw_effects["model"] = (
        "Sensitivity: raw incidence rate"
    )

    lag_effects["model"] = (
        "Sensitivity: one-year lag income"
    )

    all_results = pd.concat(
        [
            main_effects,
            raw_effects,
            lag_effects,
        ],
        ignore_index=True,
    )

    all_results.to_csv(
        OUTPUT_DIR / "all_model_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 60)
    print("解析終了")
    print("=" * 60)
    print(
        f"結果保存先："
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
