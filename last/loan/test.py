from math import asin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib as mpl
import os
import pickle
from sklearn.utils import Bunch
# 警告处理 
import warnings
warnings.filterwarnings('ignore')

from last.loan.base import lode_origin_data,clear_data,self_split
from last import tree_base 
import json

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file)

def getData():
    #显示所有列
    pd.set_option('display.max_columns',None)
    isPart = True
    cache= False
    df_train,df_test = lode_origin_data(isPart,cache)
    if not cache:
        df_train,df_test=clear_data(df_train,df_test)
    X_train, X_test, y_train, y_test = self_split(df_train)
    return X_train, X_test, y_train, y_test



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
    # try:
    #if name =="ID3":
    ##        print("y_train",y_train)   y_train [0 0 0 ... 0 0 0]
    #    tree.add_features(X_train,y_train)
    #    tree.information_gain(X_train, y_train)
    #    return tree
    #else:
    print(f"训练{name}模型 start")
    print("X_train\n",X_train)
    print("y_train\n",y_train)
    model=tree.fit(X_train,y_train)
    # model=tree.fit(X_train.drop(columns=['id'],axis=1), y_train)
    # C4.5  The accuracy of the validation on the test data set is : 0.2025 ---> 0.429
    print(f"训练{name}模型 success")    
        # 保存模型到文件
        # with open(model_path, 'wb') as file:
            # pickle.dump(model, file)    
    return model

def TestAndShowResult(name,model,X_train,X_test,y_train,y_test):
    from sklearn import metrics
    # 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = model.predict(X_train)
    # train_predict = model.predict(X_train.drop(columns=['id'],axis=1))
    # print(f"决策树：{name},预测训练集结果如下：")
    # print("train_predice=",train_predict)

    print(f"决策树：{name},预测测试集结果如下：")
    test_predict = model.predict(X_test)
    # test_predict = model.predict(X_test.drop(columns=['id'],axis=1))
    # print("test_predice=",test_predict)
    zero_list= np.array([0]*len(y_test))
    zero_rank= metrics.accuracy_score(y_test,zero_list)
    print("zero_rank=",zero_rank)
    # assert any(test_predict), "列表全为0"
    # print("列表中存在非零元素")

    # print(
    if name=="C45":
        # model.evaluate(X_test,y_test)
        model.summary()
#        import graphviz
#        model.generate_tree_diagram(graphviz,"File-Name")
        # )
    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the validation on the training data set is :',metrics.accuracy_score(y_train,train_predict))
    print('The accuracy of the validation on the test data set is :',metrics.accuracy_score(y_test,test_predict))
    
    # 绘制AUC曲线
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = getAUC(y_test, model.predict_proba(X_test)[:,1])
        print("AUC ( Area Under the ROC Curve ) =",roc_auc)
        # plot_roc_curve(y_test, model.predict_proba(X_test)[:,1])
    except:
        pass

if __name__ == '__main__':
    X_train, X_test, y_train, y_test= getData()
    names=[
#      "SKLEARN",
        "ID3",
        "CART",
        "C45",
    ]
    for name in names:
        tree=tree_base.build_tree(name,X_train,y_train)
        getModel(name,tree,X_train,y_train)
        TestAndShowResult(name,tree,X_train,X_test,y_train,y_test)