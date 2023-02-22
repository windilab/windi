import itertools
import math
import pandas as pd

df = pd.read_table("transition_shapley.csv", delimiter=",")
print(df.head())

n = 3  # int(input())  # プレイヤー数を入力

seq = [str(i + 1) for i in range(n)]  # 提携の集合を作る下準備
All_set = []
for i in range(n):  # 提携の集合(リスト)を生成
    comb = list(itertools.combinations(seq, i + 1))
    # itertools.combinationで重複無しの組み合わせを生成
    All_set.extend(comb)

new_All_set = ['0']  # 後の計算のため0人の提携を入れておく
for i in range(len(All_set)):
    # 上で生成したリストでは、各提携がタプルになっているのでstrに修正
    s = ""
    a = All_set[i]
    for j in a:
        s += j
    new_All_set.append(s)

zero_set = list(0 for _ in range(len(new_All_set)))  # 提携値すべてが0の集合(リスト)

S = new_All_set  # すべての提携の集合(リスト)
V = zero_set  # すべての提携値の集合(リスト)、このあと提携値を入力する。ここではまだ0

for i in range(len(new_All_set)):
    inp = (input().split())  # ここで提携値の入力を処理
    if inp[0] in S:  # 入力した提携が提携の集合にあれば入力処理
        position = S.index(inp[0])
        V[position] = float(inp[1])
    if inp[0] == "ZERO":
        # 入力する提携値の残りがすべて0になったらZEROと入力することでfor文から抜ける
        break

sv = []
for i in range(n):
    res = 0
    i_in = [s for s in S if str(i + 1) in s]  # iが属する提携の集合(リスト)
    i_not_in = [s for s in S if str(i + 1) not in s]  # iが属さない提携の集合(リスト)
    for j in range(len(i_in)):
        res += math.factorial(len(i_in[j]) - 1) * math.factorial(n - len(i_in[j])) / math.factorial(n) \
               * (V[S.index(i_in[j])] - V[S.index(i_not_in[j])])
    # ここでシャープレイ値を計算
    sv.append(["player" + str(i + 1), res])  # 各プレイヤーのシャープレイ値をリストにまとめる
print(sv)

for i in range(12):
    df = df[df['age_ctgr'] == i]
    # 変数定義
    X = df[['body', 'intel', 'heart']].values  # 説明変数
    y = df['val'].values  # 目的変数

# https://qiita.com/leisurely/items/4cb3fb64e487dba61b15
