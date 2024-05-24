import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化决策树分类器
clf = DecisionTreeClassifier()

# 训练决策树模型并记录训练时间
start_train_time = time.time()
clf.fit(X_train, y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time

# 预测测试集并记录预测时间
start_predict_time = time.time()
y_pred = clf.predict(X_test)
end_predict_time = time.time()
prediction_time = (end_predict_time - start_predict_time) / len(X_test)

# 计算准确率和精确率
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

# 计算AUC和绘制ROC曲线
y_score = clf.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(iris.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制ROC曲线
plt.figure()
for i in range(len(iris.target_names)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {iris.target_names[i]} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 输出结果
print(f"Training Time: {training_time:.6f} seconds")
print(f"Prediction Time: {prediction_time:.6f} seconds per sample")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
for i in range(len(iris.target_names)):
    print(f"AUC for class {iris.target_names[i]}: {roc_auc[i]:.2f}")