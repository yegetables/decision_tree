from sklearn.tree import DecisionTreeClassifier
# from demo3 import ID3,C45,CART
import C45 
# from classic_ID3_decision_tree import DecisionTreeClassifier as ID3
import pandas as pd
# from tree.C13_id3_categorical import DecisionTree as OldTree_Categotical
# from tree.C14_id3_continuous import DecisionTree as OldTree_Continuous
# from tree.C15_cart_imp import MyCART as CartTree
def build_ID3(X_train,y_train):
    #id3_2=ID3()
    #return id3_2
    return DecisionTreeClassifier(criterion='entropy')
    # return OldTree_Categotical(criterion="id3",alpha=1.25)
def build_C45(X_train,y_train):
    # return OldTree_Categotical(criterion="c45")
    # return DecisionTreeClassifier(criterion='entropy')
    # from c4dot5 import DecisionTreeClassifier as C45 
    # attributes_map = {
    #     "id": "continuous",
    #     "loanAmnt":
        
    #     "Outlook": "categorical", "Humidity": "continuous",
    #     "Windy": "boolean", "Temperature": "continuous"
    # }
    # decision_tree = DecisionTreeClassifier(attributes_map)
    # decision_tree.fit(training_dataset)
    # return C45.DecisionTreeClassifier(attributes_map,max_depth=3)
    return  C45.C45Classifier()
    # from .tempC45 import C45Classifier
    # return C45Classifier()
def build_CART(X_train,y_train):
    # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
    return DecisionTreeClassifier( criterion='gini')
    # return CartTree(min_samples_split=2, pruning=True, random_state=2024)
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