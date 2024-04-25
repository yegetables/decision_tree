# 导入所需的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from matplotlib import pyplot as plt


# 加载鸢尾花数据集 数据集内容，包括如下所示的若干子数据。 list(iris.keys()) ['data',  'target',  'frame',  'target_names',  'DESCR',
# 'feature_names', 'filename']
# 数据集特征名称 iris['feature_names'] ['sepal length (cm)',  'sepal width (cm)',  'petal length (cm)',
# 'petal width (cm)'] sepal length 花萼长度、sepal width 花萼宽度、petal length 花瓣长度、petal width 花瓣宽度

# 数据集特征值 iris['data']
# 如下所示二维数组的每一行，代表一个样本的特征值，这些特征值对应于 sepal length 花萼长度、sepal width 花萼宽度、petal length 花瓣长度、petal width 花瓣宽度
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
# 数据集类别名称iris['target_names'] array(['setosa', 'versicolor', 'virginica'], dtype='<U10') 三种植物名称：setosa、versicolor、virginica
# 数据集类别值 iris['target']
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
def tree3(clf):
    fig = plt.figure(figsize=(35, 10))
    plot_tree(clf, fontsize=8)
    current_work_dir = os.path.dirname(__file__)
    fig.savefig(os.path.join(current_work_dir, "tree-ent.png"))


iris = load_iris()
X = iris.data
y = iris.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器模型
# [1] criterion，节点分割标准，规定了该决策树所采用的的最佳分割属性的判决方法，有两种：“gini”（默认按照GINI基尼系数分割），“entropy”（使用信息熵作为划分标准）。
# [2] min_weight_fraction_leaf，限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起分割。
# https://samperson1997.github.io/2018/06/18/decision-tree/
model = DecisionTreeClassifier(criterion='gini')

# 在训练集上训练模型
# print("X_train ", X_train)
# print("y_train ", y_train)
model.fit(X_train, y_train)
tree3(model)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率：{accuracy}")

# 输出分类结果
# for i in range(len(X_test)):
#     print(f"预测类别：{y_pred[i]}, 实际类别：{y_test[i]}")
