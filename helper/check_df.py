import numpy as np
import pandas as pd
import scipy.stats as stats


def missing_data(data):
    total = data.isnull().sum()
    percent = data.isnull().sum() / data.isnull().count() * 100
    tt = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt["Types"] = types
    return np.transpose(tt)


def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    items = []
    vals = []
    for col in data.columns:
        try:
            itm = data[col].value_counts().index[0]
            val = data[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt["Most frequent item"] = items
    tt["Frequence"] = vals
    tt["Percent from total"] = np.round(vals / total * 100, 3)
    return np.transpose(tt)


def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt["Uniques"] = uniques
    return np.transpose(tt)


def anova(df: pd.DataFrame, group_names: list[str], target: str):
    """
    ANNOVA(分散分析)
        p値が 0.05 未満 → 「偶然ではなく、統計的に有意な差がある」と判断
        本当に無関係だったら、こんな差が偶然に起きる確率はほぼゼロ
        説明変数と目的変数の間には非常に強い関係があることを統計的に裏付けている
        F値が 1 に近い → グループ間の差はほぼなく、ばらつきは同じ。
        F値が 大きい（今回: 45.48） → グループ間の平均の差が、偶然では説明できないほど大きい"""
    for group_name in group_names:
        # カテゴリごとに数値のリストを作る
        groups = [group[target].values for _, group in df.groupby(group_name)]

        # 一元配置分散分析（ANOVA）
        f_stat, p_value = stats.f_oneway(*groups)
        print(group_name)
        print("F値:", f_stat)
        print("p値:", p_value)
        print()
