from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype

def plot_all_scatter(df: pd.DataFrame, columns: list[str], hue: str):
    """plot_all_scatter

    複数カラムの散布図を一括で描画する

    Attributes:
        df (pd.DataFrame):
            データフレーム
        columns (list[str]):
            散布図のx軸またはy軸となるカラム名
        hue (str):
            色分けの対象となるカラム列
    """
    # カラム名の数をカウント
    column_size = len(columns)
    # カラムの組み合わせを作成
    pairs = list(combinations(columns, 2))
    # カラムの組み合わせ数を算出
    combination = len(pairs)
    # 組み合わせ数からプロットのサイズを算出
    size = combination // 2 + combination % 2

    if column_size > 2:
        f, ax = plt.subplots(size, 2, figsize=(12, 4 * size))
        for i in range(combination):
            x = i // 2
            y = i % 2
            sns.scatterplot(data=df, x=pairs[i][0], y=pairs[i][1], hue=hue, ax=ax[x, y])
    elif column_size == 2:
        f, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.scatterplot(data=df, x=pairs[0][0], y=pairs[0][1], hue=hue)
    else:
        print("カラムは2つ以上渡してください。")

    plt.show()



def plot_all_hist(df: pd.DataFrame, columns: list[str], hue: str):
    """plot_all_scatter

    複数カラムのヒストグラムを一括で描画する

    Attributes:
        df (pd.DataFrame):
            データフレーム
        columns (list[str]):
            ヒストグラムを描画するカラム名
        hue (str):
            色分けの対象となるカラム列
    """
    column_size = len(columns)
    size = column_size // 2 + column_size % 2

    if column_size > 2:
        f, ax = plt.subplots(size, 2, figsize=(12, 4 * size))
        for i, column in enumerate(columns):
            x = i // 2
            y = i % 2
            if is_numeric_dtype(df[column]):
                for h in df[hue].unique():
                    g = sns.histplot(df.loc[df[hue]==h, column], ax=ax[x, y], label=h)
            else:
                sns.countplot(x=column, data=df, ax=ax[x, y], hue=hue)
            g.legend()
            g.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    elif column_size == 2:
        f, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, column in enumerate(columns):
            y = i % 2
            for h in df[hue].unique():
                g = sns.histplot(df.loc[df[hue]==h, column], ax=ax[y], label=h)
                g.legend()
                g.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    else:
        print("カラムは2つ以上渡してください。")
    plt.show()