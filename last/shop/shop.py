from math import asin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib as mpl
import os
from sklearn import base
from sklearn.utils import Bunch
# 警告处理 
import warnings
warnings.filterwarnings('ignore')

from tree.tree_base import build_tree
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
    # df['日期'] = df['日期'].apply(loan.base.encoder1)
    # df['日期'].value_counts()  
    df['日期'] = pd.to_datetime(df['日期'])
    # 定义参考日期
    reference_date = pd.Timestamp('1970-01-01')
    # 将日期转换为距离参考日期的天数
    df['月份'] = df['日期'].dt.month
    df['日期'] = (df['日期'] - reference_date).dt.days
    
    #浏览量--->k
    # bins = [0, 1000, 5000, 10000,50000,100000,500000,1000000,1000000000]
    # labels = ['1', '5', '10', '50', '100', '500', '1000', '10000']
    # df['浏览量区间'] = pd.cut(df['浏览量'], bins=bins, labels=labels).
    
    magnitude = 10 ** len(str(df['浏览量'].max())) - 1
    # 定义区间边界（根据数量级划分）
    bins = [10**i for i in range(len(str(df['浏览量'].max()))+1)]
    df['浏览量区间'] = pd.cut(df['浏览量'], bins=bins).apply(lambda x: (x.left + x.right) / 2).value_counts()
    
    #访客数--->k
    # bins = [0, 1000, 5000, 10000,50000,100000,500000,1000000,1000000000]
    labels = ['1', '5', '10', '50', '100', '500', '1000', '10000']
    # df['访客数区间'] = pd.cut(df['访客数'], bins=bins, labels=labels)
    magnitude = 10 ** len(str(df['访客数'].max())) - 1
    # 定义区间边界（根据数量级划分）
    bins = [10**i for i in range(len(str(df['访客数'].max()))+1)]
    df['访客数区间'] = pd.cut(df['访客数'], bins=20).apply(lambda x: (x.left + x.right) / 2).value_counts()
        
    df['人均浏览量'] = pd.cut(df['人均浏览量'], bins=20).apply(lambda x: (x.left + x.right) / 2).value_counts()
    df['平均停留时间'] = pd.cut(df['平均停留时间'], bins=20).apply(lambda x: (x.left + x.right) / 2).value_counts()
    del_clolumns=[
        "日期",
        "浏览量",
        "访客数",
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
            
    df["成交金额"]= df["成交金额"].astype(int)
    df.to_csv(current_dir+'/data/shop_process.csv',index=False)
    return df

def self_split(train):
    from sklearn.model_selection import train_test_split
    # ## 选择其类别为0和1的样本 （不包括类别为2的样本）
    # 成交金额
    # data_target_part = train["成交金额"]
    # data_features_part = train[[x for x in train.columns if x != "成交金额"]]
    data_target_part = train["成交金额"]
    data_features_part = train[[x for x in train.columns if x != "成交金额"]]
    X_train, X_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 2024)
    return X_train, X_test, y_train, y_test

def TestAndShowResult(name,model,X_train,X_test,y_train,y_test):
    from sklearn import metrics
    zero_list= np.array([0]*len(y_test))
    zero_rank= metrics.accuracy_score(y_test,zero_list)
    print("zero_rank=",zero_rank)
    # 在训练集和测试集上分布利用训练好的模型进行预测
    if name=="C45":
        xv=X_train.values
        xt=X_test.values
        train_predict=model.predict(xv)
        test_predict = model.predict(xt)
    else:
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
    if name=="C45":
        # model.summary()
        # model.evaluate(X_test,y_test)
        model.print_rules()
        pass
    
    print(f"决策树：{name},预测测试集结果如下：")
    # print("预测结果",test_predict)
    # print("实际结果",y_test)

    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the validation on the training data set is :',metrics.accuracy_score(y_train,train_predict))
    print('The accuracy of the validation on the test data set is :',metrics.accuracy_score(y_test,test_predict))
    
    # 计算均方误差
    mse = metrics.mean_squared_error(y_test,test_predict)
    print(f'均方误差：{mse}')
    # 计算平均绝对误差
    mae = metrics.mean_absolute_error(y_test,test_predict)
    print(f'平均绝对误差：{mae}')
    
    # 绘制AUC曲线
    # try:
        # from sklearn.metrics import roc_auc_score
        # roc_auc = getAUC(y_test, model.predict_proba(X_test)[:,1])
        # print("AUC ( Area Under the ROC Curve ) =",roc_auc)
        # plot_roc_curve(y_test, model.predict_proba(X_test)[:,1])
    # except:
        # pass

def getModel(name,tree,X_train,y_train):
    model_path=current_dir+"/data/"+name+".pkl"
    # if os.path.exists(model_path):
    #     # 加载模型
        # print(f"加载{name}模型 success")
        # with open(model_path, 'rb') as file:
            # model = pickle.load(file)
    # else:
        # 训练模型
    model=None
    print(f"训练{name}模型 start")
    # print("X_train\n",X_train)
    # print("y_train\n",y_train)
    if name=="C45":
        xv=X_train.values
        yv=y_train.values
        model=tree.fit(xv,yv,indexContinuousFeatures_=tuple(range(len(xv[0])))).pruning()  # 剪枝
    else:
        model=tree.fit(X_train,y_train)
    print(f"训练{name}模型 success")    
        # 保存模型到文件
        # with open(model_path, 'wb') as file:
            # pickle.dump(model, file)    
    return model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test= getData()
    names=[
#      "SKLEARN",
        "ID3",
        "CART",
        "C45",
    ]
    for name in names:
        tree=build_tree(name,X_train,y_train)
        getModel(name,tree,X_train,y_train)
        TestAndShowResult(name,tree,X_train,X_test,y_train,y_test)