import sklearn as sk
# import jieba
import pandas as pd
import joblib
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
from sklearn.linear_model import *
from sklearn.metrics import *

def linera1():
    #正规方程(过拟合）
    boston=sk.datasets.load_boston()
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    estimator=LinearRegression()
    estimator.fit(x_train,y_train)

    print("正规方程权重为：",estimator.coef_)
    print("正规方程的偏置：",estimator.intercept_)

    y_predict=estimator.predict(x_test)
    error=mean_squared_error(y_predict,y_test)
    print("正规方程的MSE：" ,error)


def linera2():
    #梯度下降
    boston = sk.datasets.load_boston()

    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000, penalty="l1")
    estimator.fit(x_train, y_train)

    print("梯度下降权重为：",estimator.coef_)
    print("梯度下降的偏置：",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_predict, y_test)
    print("梯度下降MSE：", error)


def linera3():
    #ridge
    boston = sk.datasets.load_boston()

    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge(alpha=1,max_iter=10000)
    estimator.fit(x_train, y_train)

    print("ridge权重为：",estimator.coef_)
    print("ridge偏置：",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_predict, y_test)
    print("ridge MSE：", error)


if __name__ == "__main__":
    linera1()
    print("--------------------------------------------------------------------")
    linera2()
    print("--------------------------------------------------------------------")
    linera3()