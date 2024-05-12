from sklearn.tree import DecisionTreeClassifier
def build_ID3(X_train,y_train):
    return DecisionTreeClassifier(criterion='entropy')
def build_C45(X_train,y_train):
    # import C45 
    # return  C45.C45Classifier()
    from .C45.CC4 import C45decisionTree
    return C45decisionTree(
            minSamplesLeaf=1,  # 超参数：叶结点的最少样本数量
            maxDepth=7,        # 超参数：最大树深
            maxPruity=1.,      # 超参数：叶结点的最大纯度
            # maxFeatures=4,     # 超参数：最大特征数
            α=2.5,              # 超参数：代价复杂度剪枝的惩罚参数
    )
def build_CART(X_train,y_train):
    # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
    return DecisionTreeClassifier( criterion='gini')
def build_DEFAULT(X_train,y_train):
    return build_CART(X_train,y_train)

def build_tree(func_name,X_train,y_train):
        # "SKLEARN",
        # "ID3",
        # "C45",
        # "CART"
        # "DEFAULT"
    if func_name=="ID3":
        return build_ID3(X_train,y_train)
    elif func_name=="C45":
        return build_C45(X_train,y_train)
    elif func_name=="CART":
        return build_CART(X_train,y_train)
    else:
        return build_DEFAULT(X_train,y_train)