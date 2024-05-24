import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
plt.rcParams["font.sans-serif"] = ["SimHei"]  # windows系统
plt.rcParams['axes.unicode_minus']=False      #正常显示符号


import os
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file)

def lode_origin_data():
    data1=pd.read_excel(current_dir+'/data/dianshang.xlsx',sheet_name='user_info')
    data2=pd.read_excel(current_dir+'/data/dianshang.xlsx',sheet_name='home_page').rename(columns={'page':'home_page'})
    data3=pd.read_excel(current_dir+'/data/dianshang.xlsx',sheet_name='listing_page').rename(columns={'page':'listing_page'})
    data4=pd.read_excel(current_dir+'/data/dianshang.xlsx',sheet_name='product_page').rename(columns={'page':'product_page'})
    data5=pd.read_excel(current_dir+'/data/dianshang.xlsx',sheet_name='payment_page').rename(columns={'page':'payment_page'})
    data6=pd.read_excel(current_dir+'/data/dianshang.xlsx',sheet_name='confirmation_page').rename(columns={'page':'confirmation_page'})
    data=pd.merge(data1,data2,on='user_id',how='outer')
    data=pd.merge(data,data3,on='user_id',how='outer')
    data=pd.merge(data,data4,on='user_id',how='outer')
    data=pd.merge(data,data5,on='user_id',how='outer')
    data=pd.merge(data,data6,on='user_id',how='outer')
    # data
    # data.info()
    # print(data.isnull().sum())
    return data

def clearData(data):
    data=data.dropna(subset=data.columns[:9])
    # print(data.isnull().sum())
    # #空值已经被我们剔除了

    # plt.figure(figsize=(12,4))
    # data['age'].plot(kind='box')
    # plt.title('用户年龄箱线图')
    data=data[data['age']<100]
    data.reset_index(drop=True, inplace=True)
    # data

    # 对于用户的行为状态，我们把空值全部变为0，否则变为1，代表着该用户是否有这样的行为
    data=data.fillna(0)
    data.iloc[:,9:]=data.iloc[:,9:].apply(pd.to_numeric, errors='coerce').fillna(1).astype(int)
    # data


    # 用户来源主要有三个 老用户主要来源于Direct,远大于其他两个，最少来源于Ads；新用户则最多来源于Seo，最少来源于Direct，这跟老用户截然不同。 不同的来源对于不同的用户产生不一样的影响，我们可以在分布在Direct和Seo大力采取推广手段，这样既能让老用户回流，还能吸引新用户到来
    # source=data.groupby(['new_user','source'])['source'].count().unstack()
    # plt.figure(figsize=(12,4))
    # plt.subplot(121)
    # plt.plot(source.columns,source.loc[0],label='老用户')
    # plt.plot(source.columns,source.loc[1],label='新用户')
    # plt.legend()
    # plt.show()

    # source.plot(kind='bar')
    # plt.xticks([0,1],['老用户','新用户'],rotation=0)
    # plt.show()

    # y_rank = stats.rankdata(data['payment_page'])

    # 计算斯皮尔曼相关系数及 p 值
    # corr, p_value = stats.spearmanr(data['total_pages_visited'], y_rank)

    # print("斯皮尔曼相关系数:", corr)
    # print("p 值:", p_value)
    # see_data=data[data['payment_page']==1]['total_pages_visited']
    # no_data=data[(data['payment_page']==0) &((data['product_page']==1))]['total_pages_visited']


    # fig, ax = plt.subplots()


    # 添加标题和标签
    # ax.set_title('是否进入第四阶段用户浏览页面次数密度分布图')
    # ax.set_xlabel('浏览页面的次数')
    # ax.set_ylabel('密度')

    # print('浏览过支付结算页的用户浏览页面次数平均数:',see_data.describe())
    # print('没有进入到这个阶段的用户平均数:',no_data.describe())


    #提取出payment_page为0或1的用户作为数据集
    see_no_data=data[((data['payment_page']==0) &(data['product_page']==1)) | (data['payment_page']==1)]
    # see_no_data
    see_no_data['source']=pd.Categorical(see_no_data['source']).codes.astype('int64')
    see_no_data['payment_page']=see_no_data['payment_page'].astype('int64')
    see_no_data['sex']=pd.Categorical(see_no_data['sex']).codes.astype('int64')
    
    # print(see_no_data.head())
    see_no_data.to_csv(current_dir+'/data/seenodata_process.csv',index=False)
    # see_no_data
    #0代表Ads，1代表Direct ，2代表Seo
    #0代表女性，1代表男性
    return see_no_data

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
    if name=="C45":
        xv=X_train.values
        yv=y_train.values
        start_train_time = time.time()
        model=tree.fit(xv,yv,indexContinuousFeatures_=tuple(range(len(xv[0]))))
        end_train_time = time.time()
        training_time = end_train_time - start_train_time
        # model=model.pruning()  # 剪枝
    else:
        start_train_time = time.time()
        model=tree.fit(X_train,y_train)
        end_train_time = time.time()
        training_time = end_train_time - start_train_time
                
    print(f"训练{name}模型 success,training_time={training_time}")    
        # 保存模型到文件
        # with open(model_path, 'wb') as file:
            # pickle.dump(model, file)    
    return model



if __name__ == '__main__':
    data=lode_origin_data()
    data.to_csv(current_dir+'/data/origin_process.csv',index=False)
    see_no_data=clearData(data)
    x = see_no_data[['new_user', 'age','sex','source','total_pages_visited']]
    y = see_no_data['payment_page']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)
    names=[
#      "SKLEARN",
        "ID3",
        "CART",
        "C45",
    ]
    from tree import tree_base
    for name in names:
        tree=tree_base.build_tree(name,X_train,y_train)
        getModel(name,tree,X_train,y_train)
        from last.loan.loan import TestAndShowResult
        TestAndShowResult(name,tree,X_train,X_test,y_train,y_test)
        
# 各模型的特征系数最大为total_pages_visited，即浏览页面次数，其次为年龄；而是否为新用户,性别以及来源都很小程度的影响第四阶段转化率
# 浏览次数越多，年龄越大的用户，越有可能进入到支付页面


# 该电商的老用户多于新用户
# 女性用户多于男性用户
# 年龄为20-40岁区间居多
# 老用户来源于Direct最多，新用户来源于Seo最多。
# 浏览总数大概在6次左右的用户居多
# 如果想吸引新用户，可以在Seo上多做推广，多向女性用户宣传推广，浏览页面的设计应该多向中年人的审美靠近，并且增添特色吸引用户继续了解，从而增加浏览总数
# 该商品在“浏览产品详情页” 到 “浏览支付页面” 这一阶段的转化率出现了问题，分析发现跟浏览页面总数和年龄有一定的关系
# 我们需要提高用户的浏览总数，吸引大龄用户更能提高该阶段的转化率。

#数据集https://tianchi.aliyun.com/dataset/154063
# user_id ：用户id
# new_user ：是否新用户 是：1、否：0
# age ：用户年龄
# sex ：用户性别
# market ：用户所在市场级别
# device ：用户设备
# operative_system ：操作系统
# source ：来源
# total_pages_visited ：浏览页面总数
# home_page : 浏览过主页的用户
# listing_page : 浏览过列表页的用户
# product_page : 浏览过产品详情页的用户
# payment_page : 浏览过支付结算页的用户
# payment_confirmation_page : 浏览过确认支付完成页的用户