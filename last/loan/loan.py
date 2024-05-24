from calendar import c
from webbrowser import get
import pandas as pd
import numpy as np
import time
import os
# 警告处理 
import warnings
warnings.filterwarnings('ignore')

from last.loan.loan_base import lode_origin_data,clear_data,self_split
from tree import tree_base
import json
from sklearn import metrics
import matplotlib.pyplot as plt

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file)

def getData():
    #显示所有列
    pd.set_option('display.max_columns',None)
    isPart = True
    cache= False
    num=100000
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

# 计算预测结果 精确率 Precision
def getPrecision(y_test, test_predict):
    precision_binary = metrics.precision_score(y_test, test_predict, average='binary')
    # print(f"Precision-binary: {precision_binary:.2f}")
    precision_weighted = metrics.precision_score(y_test, test_predict, average='weighted')
    # print(f"Precision-weighted: {precision_weighted:.2f}")
    precision_micro = metrics.precision_score(y_test, test_predict, average='micro')
    # print(f"Precision-micro: {precision_micro:.2f}")
    precision_macro = metrics.precision_score(y_test, test_predict, average='macro')
    # print(f"Precision-macro: {precision_macro:.2f}")
    # 二分类问题
    # 'binary'：
    # 适用于二分类问题。
    # 默认情况下，precision_score 在二分类问题中会使用这个设置。
    # 'micro'：
    # 适用于二分类和多分类问题。
    # 计算全局精确率，将所有的真正例（TP）、假正例（FP）、假负例（FN）累加后再计算精确率。
    # 更关注整体的分类性能，忽略类别的不平衡。
    # 'macro'：
    # 适用于多分类问题，但在二分类问题中也可以使用。
    # 计算每个类别的精确率，然后对它们进行简单平均。
    # 每个类别的贡献相等，忽略类别的不平衡。
    # 'weighted'：
    # 适用于多分类问题，但在二分类问题中也可以使用。
    # 计算每个类别的精确率，并根据每个类别的样本数量进行加权平均。
    # 适用于类别不平衡的情况。
    # pass
    print(f"Precision: {precision_weighted:.2f}")
    return precision_weighted

# 计算准确率 accuracy
def getAccuracy(y_test, test_predict):
    # 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    # print('The accuracy of the validation on the training data set is :',metrics.accuracy_score(y_train,train_predict))
    accuracy_test = metrics.accuracy_score(y_test, test_predict)
    print(f"Accuracy: {accuracy_test:.2f}")
    return accuracy_test

def getF1(y_test, test_predict):
    f1 = metrics.f1_score(y_test, test_predict, average='weighted')
    print(f"F1 Score: {f1:.2f}")
    return f1

def getRecall(y_test, test_predict):
    recall = metrics.recall_score(y_test, test_predict, average='weighted')
    print(f"Recall: {recall:.2f}")
    return recall

def getMatrix(y_test, test_predict):
    conf_matrix = metrics.confusion_matrix(y_test, test_predict)
    print("Confusion Matrix:")
    print(conf_matrix)
    return conf_matrix


def getAUC(model,X_test,y_test,target_names):
    # 计算AUC和绘制ROC曲线
    y_score = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test == i, y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    plt.figure()
    for i in range(len(target_names)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {target_names[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    for i in range(len(target_names)):
        print(f"AUC for class {target_names[i]}: {roc_auc[i]:.2f}")
    # ROC 曲线：ROC 曲线是以假阳性率 (FPR) 为横轴，真阳性率 (TPR) 为纵轴的曲线，用于评估分类模型在不同阈值下的性能。
    # AUC (Area Under the Curve)：AUC 是 ROC 曲线下的面积，表示分类器在不同阈值下的性能。AUC 值越接近 1，模型性能越好。
    pass

def TestAndShowResult(name,model,X_train,X_test,y_train,y_test):
    xt=X_test
    # 在测试集上进行预测
    if name=="C45":
        xt=X_test.values
    start_predict_time = time.time()
    test_predict = model.predict(xt)
    end_predict_time = time.time()
    prediction_time = (end_predict_time - start_predict_time) / len(xt)
    print(f"Prediction Time: {prediction_time:.10f} seconds per sample")
    
    getAccuracy(y_test, test_predict)
    getPrecision(y_test, test_predict)
    getF1(y_test, test_predict)
    getRecall(y_test, test_predict)
    getMatrix(y_test, test_predict)
    if name !="C45":
        getAUC(model,xt,y_test,[0,1])
    if name!="C45":
        # 将特征名称和重要性值按顺序配对打印
        for feature_name, importance in zip(xt.columns, model.feature_importances_):
            print(f"Feature: {feature_name}, Importance: {importance}")
if __name__ == '__main__':
    training_time=0.0
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