import pandas as pd
import numpy as np
import os
# 警告处理 
import warnings
warnings.filterwarnings('ignore')

from last.loan.loan_base import lode_origin_data,clear_data,self_split,getAUC,plot_roc_curve
from tree import tree_base
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
    num=1000
    df_train = lode_origin_data(isPart,cache,num)
    if not cache:
        df_train=clear_data(df_train)
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
    print(f"训练{name}模型 start")
    # print("X_train\n",X_train)
    # print("y_train\n",y_train)
    if name=="C45":
        xv=X_train.values
        yv=y_train.values
        model=tree.fit(xv,yv,indexContinuousFeatures_=tuple(range(len(xv[0]))))
        model=model.pruning()  # 剪枝
    else:
        model=tree.fit(X_train,y_train)
    print(f"训练{name}模型 success")    
        # 保存模型到文件
        # with open(model_path, 'wb') as file:
            # pickle.dump(model, file)    
    return model

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
    if name!="C45":
        # 将特征名称和重要性值按顺序配对打印
        for feature_name, importance in zip(X_train.columns, model.feature_importances_):
            print(f"Feature: {feature_name}, Importance: {importance}")
    
    # 绘制AUC曲线
    # try:
        # from sklearn.metrics import roc_auc_score
        # roc_auc = getAUC(y_test, model.predict_proba(X_test)[:,1])
        # print("AUC ( Area Under the ROC Curve ) =",roc_auc)
        # plot_roc_curve(y_test, model.predict_proba(X_test)[:,1])
    # except:
        # pass

if __name__ == '__main__':
    X_train, X_test, y_train, y_test= getData()
    names=[
#      "SKLEARN",
        "ID3",
        "CART",
        "C45",
    ]
    for name in names:
        # 根据参数不同构建不同算法的决策树
        tree=tree_base.build_tree(name,X_train,y_train)
        # 训练模型
        getModel(name,tree,X_train,y_train)
        # 测试模型性能并展示结果
        TestAndShowResult(name,tree,X_train,X_test,y_train,y_test)