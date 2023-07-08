import pandas as pd
import codecs
from scipy.stats import ranksums, norm
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


def spearmanr_confidence_interval(x, y, alpha=0.05):
    r = spearmanr(x, y)[0]
    n = len(x)
    se = 1 / np.sqrt(n - 3)
    z = norm.ppf(1 - alpha / 2)
    lower_bound = np.tanh(np.arctanh(r) - z * se)
    upper_bound = np.tanh(np.arctanh(r) + z * se)
    return lower_bound, upper_bound


with codecs.open("FEH_mitsudo.csv", "r", "Shift-JIS", "ignore") as file:
    df = pd.read_table(file, delimiter=",")

df["year"] = df["time_code"] / 1000000
print(df)

df = df.rename(columns={'地域2010': 'kanji', 'value': 'population_density'})
df["year"] = df['year'].astype('int')  # 人口密度のデータフレーム
# df = df[df.year.isin([1990, 2010])]
df = df[["year", "kanji", "population_density"]]
print(df.head())

df = df.set_index(["kanji"])

print(df.head())
data1 = df[df.year == 1990]["population_density"]
data2 = df[df.year == 2010]["population_density"]
print(data1)
print(data2)

corr, p_value = spearmanr(data1, data2)
lower_bound, upper_bound = spearmanr_confidence_interval(data1, data2)

print("Spearman's correlation coefficient:", corr)
print("p-value:", p_value)
print("95% Confidence Interval:", (lower_bound, upper_bound))

sns.relplot(x="year", y="population_density",
            data=df, kind="line",
            hue="kanji", style="kanji",
            markers=True, dashes=False)
# plt.show()

change_proportion = df[df.year == 2010]["population_density"] / df[df.year == 1990]["population_density"]
print(change_proportion)
print(change_proportion.describe())
