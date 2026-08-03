from pathlib import Path

import pandas as pd


# ============================================================
# 1. ファイルパス
# ============================================================

BASE_DIR = Path(
    r"C:\Users\sawai\PycharmProjects\gbd_gender\gender"
)

GBD_FILE = (
    BASE_DIR / "IHME-GBD_2019_DATA_47P_15-39.csv"
)

ESTAT_FILE = (
    BASE_DIR / "male_female_estat_.csv"
)

OUTPUT_FILE = BASE_DIR / "data.csv"


# e-Statでは所定内給与額が「千円」単位で格納されています
MALE_INCOME_COL = (
    "F620201_所定内給与額（男）（〜2019）【千円】"
)

FEMALE_INCOME_COL = (
    "F620202_所定内給与額（女）（〜2019）【千円】"
)


# GBDファイルの都道府県表記に合わせます
PREFECTURE_JP_TO_EN = {
    "北海道": "Hokkaidō",
    "青森県": "Aomori",
    "岩手県": "Iwate",
    "宮城県": "Miyagi",
    "秋田県": "Akita",
    "山形県": "Yamagata",
    "福島県": "Fukushima",
    "茨城県": "Ibaraki",
    "栃木県": "Tochigi",
    "群馬県": "Gunma",
    "埼玉県": "Saitama",
    "千葉県": "Chiba",
    "東京都": "Tōkyō",
    "神奈川県": "Kanagawa",
    "新潟県": "Niigata",
    "富山県": "Toyama",
    "石川県": "Ishikawa",
    "福井県": "Fukui",
    "山梨県": "Yamanashi",
    "長野県": "Nagano",
    "岐阜県": "Gifu",
    "静岡県": "Shizuoka",
    "愛知県": "Aichi",
    "三重県": "Mie",
    "滋賀県": "Shiga",
    "京都府": "Kyōto",
    "大阪府": "Ōsaka",
    "兵庫県": "Hyōgo",
    "奈良県": "Nara",
    "和歌山県": "Wakayama",
    "鳥取県": "Tottori",
    "島根県": "Shimane",
    "岡山県": "Okayama",
    "広島県": "Hiroshima",
    "山口県": "Yamaguchi",
    "徳島県": "Tokushima",
    "香川県": "Kagawa",
    "愛媛県": "Ehime",
    "高知県": "Kōchi",
    "福岡県": "Fukuoka",
    "佐賀県": "Saga",
    "長崎県": "Nagasaki",
    "熊本県": "Kumamoto",
    "大分県": "Ōita",
    "宮崎県": "Miyazaki",
    "鹿児島県": "Kagoshima",
    "沖縄県": "Okinawa",
}


# ============================================================
# 2. GBD発症率データの抽出
# ============================================================

def load_incidence(
    file_path: Path,
) -> pd.DataFrame:
    """
    15～39歳の統合失調症発症率を抽出する。
    """

    gbd = pd.read_csv(file_path)

    required_cols = {
        "measure",
        "location",
        "sex",
        "age",
        "cause",
        "metric",
        "year",
        "val",
    }

    missing_cols = required_cols.difference(
        gbd.columns
    )

    if missing_cols:
        raise KeyError(
            "GBDファイルに以下の列がありません："
            f"{sorted(missing_cols)}"
        )

    incidence = gbd.loc[
        (gbd["measure"] == "Incidence")
        & (gbd["age"] == "15 to 39")
        & (gbd["cause"] == "Schizophrenia")
        & (gbd["metric"] == "Rate"),
        [
            "location",
            "sex",
            "year",
            "val",
        ],
    ].copy()

    incidence = incidence.rename(
        columns={
            "location": "prefecture",
            "val": "incidence_rate",
        }
    )

    incidence["year"] = pd.to_numeric(
        incidence["year"],
        errors="raise",
    ).astype(int)

    incidence["incidence_rate"] = pd.to_numeric(
        incidence["incidence_rate"],
        errors="raise",
    )

    return incidence


# ============================================================
# 3. 所得データの抽出
# ============================================================

def load_income(
    file_path: Path,
) -> pd.DataFrame:
    """
    男女別の所定内給与額を抽出し、
    wide形式からlong形式に変換する。
    """

    usecols = [
        "調査年",
        "地域",
        MALE_INCOME_COL,
        FEMALE_INCOME_COL,
    ]

    income_wide = pd.read_csv(
        file_path,
        usecols=usecols,
    )

    # 「1990年度」から1990を取り出します
    income_wide["year"] = (
        income_wide["調査年"]
        .astype("string")
        .str.extract(
            r"(\d{4})",
            expand=False,
        )
        .astype("Int64")
    )

    # 都道府県名をGBDと同じアルファベット表記にします
    income_wide["prefecture"] = (
        income_wide["地域"].map(
            PREFECTURE_JP_TO_EN
        )
    )

    unknown_prefectures = sorted(
        income_wide.loc[
            income_wide["prefecture"].isna(),
            "地域",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown_prefectures:
        raise ValueError(
            "英語名に変換できない都道府県があります："
            + ", ".join(unknown_prefectures)
        )

    # 男女別の2列を1つのsex列にまとめます
    income = income_wide.melt(
        id_vars=[
            "prefecture",
            "year",
        ],
        value_vars=[
            MALE_INCOME_COL,
            FEMALE_INCOME_COL,
        ],
        var_name="income_source",
        value_name="income_thousand_yen",
    )

    income["sex"] = income[
        "income_source"
    ].map(
        {
            MALE_INCOME_COL: "Male",
            FEMALE_INCOME_COL: "Female",
        }
    )

    income["income_thousand_yen"] = (
        pd.to_numeric(
            income["income_thousand_yen"],
            errors="coerce",
        )
    )

    # 千円から円に変換します
    income["income"] = (
        income["income_thousand_yen"]
        * 1_000
    )

    return income[
        [
            "prefecture",
            "sex",
            "year",
            "income",
        ]
    ]


# ============================================================
# 4. データの結合と確認
# ============================================================

def validate_and_merge(
    incidence: pd.DataFrame,
    income: pd.DataFrame,
) -> pd.DataFrame:

    key_cols = [
        "prefecture",
        "sex",
        "year",
    ]

    # GBDデータの重複確認
    if incidence.duplicated(
        key_cols
    ).any():
        raise ValueError(
            "GBDデータに都道府県×性別×年の"
            "重複があります。"
        )

    # 所得データの重複確認
    if income.duplicated(
        key_cols
    ).any():
        raise ValueError(
            "所得データに都道府県×性別×年の"
            "重複があります。"
        )

    # 共通する都道府県×性別×年だけを結合します
    # 今回は1990～2019年になります
    merged = incidence.merge(
        income,
        on=key_cols,
        how="inner",
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq(
        "both"
    ).all():
        raise ValueError(
            "2つのファイル間で一致しない行があります。"
        )

    merged = merged.drop(
        columns="_merge"
    )

    merged = merged[
        [
            "prefecture",
            "sex",
            "year",
            "incidence_rate",
            "income",
        ]
    ]

    merged = (
        merged
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    # 結合キーの欠損確認
    if merged[key_cols].isna().any().any():
        raise ValueError(
            "結合キーに欠損値があります。"
        )

    # 発症率と所得の欠損確認
    analysis_cols = [
        "incidence_rate",
        "income",
    ]

    if merged[
        analysis_cols
    ].isna().any().any():

        missing_counts = (
            merged[analysis_cols]
            .isna()
            .sum()
            .to_dict()
        )

        raise ValueError(
            "解析変数に欠損値があります："
            f"{missing_counts}"
        )

    print("結合が完了しました")
    print(
        f"行数：{len(merged):,}"
    )
    print(
        "都道府県数："
        f"{merged['prefecture'].nunique()}"
    )
    print(
        "性別数："
        f"{merged['sex'].nunique()}"
    )
    print(
        "対象年："
        f"{merged['year'].min()}"
        "～"
        f"{merged['year'].max()}"
    )

    return merged


# ============================================================
# 5. 実行
# ============================================================

def main() -> None:

    incidence = load_incidence(
        GBD_FILE
    )

    income = load_income(
        ESTAT_FILE
    )

    data = validate_and_merge(
        incidence,
        income,
    )

    # Excelで開いても文字化けしにくい形式です
    data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"保存先：{OUTPUT_FILE}"
    )

    print(
        data.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
