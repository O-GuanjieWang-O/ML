import sklearn as sk
# import jieba
import pandas as pd
import time
from sklearn import *
from sklearn.model_selection import *
from sklearn.feature_extraction.text import *
from sklearn.feature_extraction import DictVectorizer
# 特征预处理
from sklearn.preprocessing import *
#数据降维 特征选择
from sklearn.feature_selection import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

def knn_iris():
    data=sk.datasets.load_iris();

    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, random_state=22)


    transfer = StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator=KNeighborsClassifier(n_neighbors=2)
    estimator.fit(x_train,y_train)

    predict=estimator.predict(x_test)
    print("predict: \n",predict)
    print(y_test==predict,"\n")
    score= estimator.score(x_test,y_test)
    print("score: ", score)

#网格搜索+交叉测试
def knn_iris_gscv():
    data = sk.datasets.load_iris();

    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier()
    param_dict={"n_neighbors":[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)

    estimator.fit(x_train, y_train)

    predict = estimator.predict(x_test)

    print("predict: \n", predict)
    print(y_test == predict, "\n")
    score = estimator.score(x_test, y_test)
    print("score: ", score)

    print("best_params_ ",estimator.best_params_,"\n")
    print("best_estimator_ ",estimator.best_estimator_,"\n")
    print("best_score_ ", estimator.best_score_, "\n")
    print("best_index_ ", estimator.best_index_, "\n")

def nb_news():
    #朴素贝叶斯
    print("read data\n")
    news=sk.datasets.fetch_20newsgroups(data_home="data",subset="all")
    print("assign dataset\n")
    x_train,x_test,y_train,y_test=train_test_split(news.data,news.target)
    print("one-hot generate features\n")
    transfer=TfidfVectorizer()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    print("init the model\n")
    estimator=MultinomialNB()
    estimator.fit(x_train,y_train)
    print("predict\n")
    predict=estimator.predict(x_test)
    print("result\n",y_test == predict, "\n")
    score = estimator.score(x_test, y_test)
    print("score: ", score)
    return None

def decision_tree_iris():
    #get dataset
    data=sk.datasets.load_iris()
    #split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, random_state=22)
    #
    estimator=DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)

    predict = estimator.predict(x_test)

    print("predict: \n", predict)
    print(y_test == predict, "\n")
    score = estimator.score(x_test, y_test)
    print("score: ", score)
    tree.export_graphviz(estimator,out_file="tree.dot",feature_names=data.feature_names)

def decision_titanic():
    data=pd.read_csv("data/titanic.csv")


if __name__ == "__main__":
    #knn_iris()
    start_time = time.time()
    print("decision_tree")
    decision_tree_iris()
    print("--------------------------------------------\n")
    print("KNN")
    knn_iris()
    print("\n--- %s seconds ---" % (time.time() - start_time))