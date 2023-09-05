import math

import pandas as pd
import numpy as np
import scipy.stats as stats
import codecs


with codecs.open("C:/Users/sawai/PycharmProjects/windi/gender/male_female_estat_.csv", "r", "UTF-8",
                 "ignore") as file:
    df1 = pd.read_table(file, delimiter=",")

print(df1.head())
