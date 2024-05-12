import pandas as pd
import matplotlib.pyplot as plt
import os
# 警告处理 
import warnings
warnings.filterwarnings('ignore')

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file)

# 读取原始数据
def lode_origin_data(isPart,cache=False,num=0):
    from tools.trimbigcsv import trimbigcsv 
    input=current_dir+"/data/train.csv"
    out=current_dir+"/data/train_part.csv"
    if isPart and os.path.exists(current_dir+'/data/train.csv')and num!=0:
        trimbigcsv(input, out, num)
    cache_train_file = current_dir+'/data/train_process.csv'
    if cache and os.path.exists(cache_train_file):
        train_file=cache_train_file
    else:
        if isPart and os.path.exists(current_dir+'/data/train_part.csv') :
            train_file=current_dir+'/data/train_part.csv'
        else:
            train_file=current_dir+'/data/train.csv'
    df_train = pd.read_csv(train_file)

    print(f"load data,cache={cache},isPart={isPart}")
    print("read ",train_file, df_train.shape)

    return df_train

# def analysize_data(df_train,df):
#     # 非数值型
#     non_numeric_cols = [
#         'grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine'
#     ]
#     # 数值型
#     numeric_cols = [
#         x for x in df_test.columns if x not in non_numeric_cols + ['isDefault']
#     ]
#     print("non_numeric_cols",non_numeric_cols)
#     print("numeric_cols",numeric_cols)
    
#     # 画箱式图
#     column = numeric_cols # 列表头
#     fig = plt.figure(figsize=(20, 40))  # 指定绘图对象宽度和高度
#     for i in range(len(column)):
#         plt.subplot(13, 4, i + 1)  # 13行3列子图
#         sns.boxplot(df[column[i]], orient="v", width=0.5)  # 箱式图
#         plt.ylabel(column[i], fontsize=8)
#     plt.show()

#     continuous_cols = [
#         'id', 'loanAmnt', 'interestRate', 'installment', 'employmentTitle', 'homeOwnership',
#         'annualIncome', 'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years',
#         'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec', 'revolBal', 'revolUtil','totalAcc',
#         'title', 'n14'
#     ] + [f'n{i}' for i in range(11)] 
#     non_continuous_cols = [
#         x for x in numeric_cols if x not in continuous_cols
#     ]

#     dist_cols = 6
#     dist_rows = len(df_test[continuous_cols].columns)
#     plt.figure(figsize=(4*dist_cols,4*dist_rows))

#     i=1
#     for col in df_test[continuous_cols].columns:
#         ax=plt.subplot(dist_rows,dist_cols,i)
#         ax = sns.kdeplot(df_train[continuous_cols][col], color="Red", shade=True)
#         ax = sns.kdeplot(df_test[continuous_cols][col], color="Blue", shade=True)
#         ax.set_xlabel(col)
#         ax.set_ylabel("Frequency")
#         ax = ax.legend(["train","test"])
        
#         i+=1
#     plt.show()


#     train_cols = 6
#     train_rows = len(df[continuous_cols].columns)
#     plt.figure(figsize=(4*train_cols,4*train_rows))

#     i=0
#     for col in df[continuous_cols].columns:
#         i+=1
#         ax=plt.subplot(train_rows,train_cols,i)
#         sns.distplot(df[continuous_cols][col],fit=stats.norm)
#         i+=1
#         ax=plt.subplot(train_rows,train_cols,i)
#         res = stats.probplot(df[continuous_cols][col], plot=plt)
#     plt.show()


#     for i in range(len(non_continuous_cols)):
#         print("%s这列的非连续性数据的分布："%non_continuous_cols[i])
#         print(df[non_continuous_cols[i]].value_counts())

#     for i in range(len(non_numeric_cols)):
#         print("%s这列非数值型数据的分布：\n"%non_numeric_cols[i])
#         print(df[non_numeric_cols[i]].value_counts())

#     # 描述性统计分析的操作。具体而言，describe()方法会计算该列的统计指标，包括计数、均值、标准差、最小值、25%分位数、中位数（50%分位数）、75%分位数和最大值。
#     df['policyCode'].describe()


# 处理数据
def clear_data(df_train):
    df_train['train_test'] = 'train'
    # 需要处理的列名
    is_na_cols = [
    'employmentTitle', 'employmentLength', 'postCode', 'dti', 'pubRecBankruptcies',
    'revolUtil', 'title',] + [f'n{i}' for i in range(15)]

    # 对缺失值 用众数填充
    for i in range(len(is_na_cols)):
        most_num_train = df_train[is_na_cols[i]].value_counts().index[0]
        df_train[is_na_cols[i]] = df_train[is_na_cols[i]].fillna(most_num_train)  
        
        # most_num_test = df_test[is_na_cols[i]].value_counts().index[0]
        # df_test[is_na_cols[i]] = df_test[is_na_cols[i]].fillna(most_num_test)

    # df_train = df[df['train_test'] == 'train']
    # df_test = df[df['train_test'] == 'test']
    # df = pd.concat([df_train, df_test], ignore_index=True)
    df=df_train
    df.reset_index(inplace=True)
    df.drop('index',inplace=True,axis=1)

    # del df_train['train_test']
    # del df_test['train_test']
    # print("merge and fill common number, info",df.info())
    # print("clear1 off ",df_train.shape,df_test.shape)

    # del df_test['isDefault']

    # 描述性统计分析的操作。具体而言，describe()方法会计算该列的统计指标，包括计数、均值、标准差、最小值、25%分位数、中位数（50%分位数）、75%分位数和最大值。
    # df['policyCode'].describe()
    # 字段只有一个值，不用了
    df.drop('policyCode',axis=1,inplace=True)

    df['n13'] = df['n13'].apply(lambda x: 1 if x not in [0] else x)
    df['n13'].value_counts()

    # 非数值型编码
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['grade'] = le.fit_transform(df['grade'])
    df['grade'].value_counts()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['subGrade'] = le.fit_transform(df['subGrade'])
    df['subGrade'].value_counts()


    df['employmentLength'] = df['employmentLength'].apply(encoder)
    df['employmentLength'].value_counts()

    df['issueDate'] = df['issueDate'].apply(encoder1)
    df['issueDate'].value_counts()  

    df['earliesCreditLine'] = df['earliesCreditLine'].apply(encoder2)
    df['earliesCreditLine'].value_counts()

    df['earliesCreditLine'] = df['earliesCreditLine'].apply(encoder1)
    df['earliesCreditLine'].value_counts()





    train = df[df['train_test'] == 'train']
    train['isDefault'] = train['isDefault'].astype('int64')
    del train['train_test']
    del train['id']
    # del train['Unnamed: 0']
    # del test['Unnamed: 0']
    # print("clear2 off ",train.shape,test.shape)
    # train = train.iloc[:, 1:]
    # test = test.iloc[:, 1:]
    # print("train columns",train.columns)
    # print("test columns",test.columns)
    train.to_csv(current_dir+'/data/train_process.csv',index=False)
    return train

# 构造编码函数
def encoder(x):
    if x[:-5] == '10+ ':
        return 10
    elif x[:-5] == '< 1':
        return 0
    else:
        return int(x[0])

from datetime import datetime
def encoder1(x):
    x = str(x)
    now = datetime.strptime('2020-07-01','%Y-%m-%d')
    past = datetime.strptime(x,'%Y-%m-%d')
    period = now - past
    period = period.days
    return round(period / 30, 2)

def encoder2(x):
    if x[:3] == 'Jan':
        return x[-4:] + '-' + '01-01'
    if x[:3] == 'Feb':
        return x[-4:] + '-' + '02-01'
    if x[:3] == 'Mar':
        return x[-4:] + '-' + '03-01'
    if x[:3] == 'Apr':
        return x[-4:] + '-' + '04-01'
    if x[:3] == 'May':
        return x[-4:] + '-' + '05-01'
    if x[:3] == 'Jun':
        return x[-4:] + '-' + '06-01'
    if x[:3] == 'Jul':
        return x[-4:] + '-' + '07-01'
    if x[:3] == 'Aug':
        return x[-4:] + '-' + '08-01'
    if x[:3] == 'Sep':
        return x[-4:] + '-' + '09-01'
    if x[:3] == 'Oct':
        return x[-4:] + '-' + '10-01'
    if x[:3] == 'Nov':
        return x[-4:] + '-' + '11-01'
    if x[:3] == 'Dec':
        return x[-4:] + '-' + '12-01'


# 数据处理
import numpy as np
import pandas as pd

# 数据可视化
import matplotlib.pyplot as plt

# 特征选择和编码
from sklearn.preprocessing import LabelEncoder

# 机器学习
from sklearn import model_selection, tree, preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# 网格搜索、随机搜索
import scipy.stats as st
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# 模型度量（分类）
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# 警告处理 
import warnings
warnings.filterwarnings('ignore')

# 在Jupyter上画图
# %matplotlib inline


# 绘制AUC曲线
def getAUC(y_test, preds):
    fpr, tpr, _ = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

import time
def plot_roc_curve(y_test, preds):
    fpr, tpr, _ = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



def self_split(train):
    # # ## 为了正确评估模型性能，将训练数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
    from sklearn.model_selection import train_test_split
    # ## 选择其类别为0和1的样本 （不包括类别为2的样本）
    data_target_part = train['isDefault']
    data_features_part = train[[x for x in train.columns if x != 'isDefault' and 'id']]
    # ## 测试集大小为20%， 80%/20%分
    # # random_state参数是用来设置随机种子
    X_train, X_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 2024)
    # print(X_train.head())
    # print(y_train.head())
    # print(X)
    return X_train, X_test, y_train, y_test