from math import asin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib as mpl
import os
import pickle
from sklearn import base
from sklearn.utils import Bunch
# 警告处理 
import warnings
warnings.filterwarnings('ignore')

from last import tree_base,loan
import json
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file)


def getData():
    #显示所有列
    pd.set_option('display.max_columns',None)
    cache= False
    df_train = lode_origin_data(cache)
    if not cache:
        df_train=clear_data(df_train)
    X_train, X_test, y_train, y_test = self_split(df_train)
    return X_train, X_test, y_train, y_test

# 读取原始数据
def lode_origin_data(cache=False):
    cache_train_file = current_dir+'/data/shop_process.csv'
    if cache and os.path.exists(cache_train_file):
        train_file=cache_train_file
    else:
            train_file=current_dir+'/data/shop.csv'
    df_train = pd.read_csv(train_file, encoding='utf-8')
    print(f"load data,cache={cache}")
    print("read ",train_file, df_train.shape)
    return df_train

# 处理数据
def clear_data(df):
    # TODO: 数据处理
    df['日期'] = df['日期'].apply(loan.base.encoder1)
    df['日期'].value_counts()  
    del_clolumns=[
        "下单成交转化率环比",
        "下单转化率环比",
        "下单商品件数环比",
        "下单金额环比",
        "下单单量环比",
        "下单客户数环比",
        "成交转化率环比",
        "成交商品件数环比",
        "客单价环比",
        "成交金额环比",
        "成交单量环比",
        "成交客户数环比",
        "跳失率环比",
        "平均停留时间环比",
        "人均浏览量环比",
        "访客数环比",
        "浏览量环比",
    ]
    for col in del_clolumns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df.to_csv(current_dir+'/data/shop_process.csv',index=False)
    return df

def self_split(train):
    from sklearn.model_selection import train_test_split
    # ## 选择其类别为0和1的样本 （不包括类别为2的样本）
    # 成交金额
    data_target_part = train["成交金额"]
    data_features_part = train[[x for x in train.columns if x != "成交金额"]]
    
    X_train, X_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 2024)
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    X_train, X_test, y_train, y_test= getData()
    names=[
#      "SKLEARN",
        "ID3",
        "CART",
        "C45",
    ]
    from last.loan.test import TestAndShowResult,getModel
    for name in names:
        tree=tree_base.build_tree(name,X_train,y_train)
        getModel(name,tree,X_train,y_train)
        TestAndShowResult(name,tree,X_train,X_test,y_train,y_test)