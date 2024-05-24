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
        
    df=df_train
    df.reset_index(inplace=True)
    df.drop('index',inplace=True,axis=1)

    # 描述性统计分析的操作。具体而言，describe()方法会计算该列的统计指标，包括计数、均值、标准差、最小值、25%分位数、中位数（50%分位数）、75%分位数和最大值。
    # df['policyCode'].describe(）
    
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


    # 标签编码（Label Encoding），将分类数据转换为数值数据
    df['employmentLength'] = df['employmentLength'].apply(encoder)
    # 填充列中类别值的出现次数。
    df['employmentLength'].value_counts()

    # 应用自定义编码函数转换日期
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




def self_split(train):
    from sklearn.model_selection import train_test_split
    # # ## 为了正确评估模型性能，将训练数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
    # ## 划分训练目标特征和数据其他特征
    data_target_part = train['isDefault']
    data_features_part = train[[x for x in train.columns if x != 'isDefault' and 'id']]
    # ## 测试集大小为20%， 80%/20%分
    # # random_state参数是用来设置随机种子
    X_train, X_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 2024)
    return X_train, X_test, y_train, y_test