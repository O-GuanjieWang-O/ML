import sklearn as sk
# import jieba
import pandas as pd
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



def data_demo():
    iris = sk.datasets.load_iris()
    print("data", iris.data)
    print("descr", iris.DESCR)
    print("feature", iris.feature_names)
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2)
    # x_train 训练集特征
    # x_test 测试集特征
    # y_train 训练集目标值
    # y_test 测试集特征
    print(x_train.shape)
    return None


"""
字典类别抽取
1. 类别特征较多
2. 字典类型（DictVectorizer 转换）
3. 本省就是字典类型
4. 行是样本，列是特征
"""


def dict_demo():
    """dict features extract"""
    # 三个样本，4个特征
    data = [{"city": "beijing", "temp": 100},
            {"city": "shanghai", "temp": 60},
            {"city": "guangzhou", "temp": 30}
            ]
    # sparse -->稀疏-->将非0值的位置（x,y) 表示出来，节省内存
    transfer = DictVectorizer(sparse=False)
    # 实例化转换器(将dict 转换为one-hot 编码（0/1）
    data_vector = transfer.fit_transform(data)

    print("feature name\n", transfer.get_feature_names())
    print("data\n", data_vector)
    return None


"""
文本特征提取
    1.单词作为特征
    2.特征：单词
方法：
    CountVectorizer-->统计特征词出现的次数
"""


def text_demo():
    text = ["life is short, i like python",
            "life is too long, i dislike python"]
    transfer = CountVectorizer("""stop_words={"like"}""")
    data_vector = transfer.fit_transform(text)
    print("feature:\n", transfer.get_feature_names(), "\n")
    print(text, "\n")
    print("data:\n", data_vector.toarray())


def text_demo_chinese():
    text = ["我爱北京天安门", "天安门上太阳升"]
    textnew = []
    for sen in text:
        temp = cut_words(sen)
        textnew.append(temp)

    transfer = CountVectorizer()
    data_vector = transfer.fit_transform(textnew)
    print("feature:\n", transfer.get_feature_names(), "\n")
    print(textnew, "\n")
    print("data:\n", data_vector.toarray())


def cut_words(text):
    t = " ".join(list(jieba.cut(text)))
    return t


"""
Tf-idf 文本特征提取
TfidfVectorizer
主要思想： 如果某个词或短语在一篇文章中出现的概率高，并在其他文章中很少出现，则认为此词或短语有很好的类别分类能力
作用： 用以评估一字词对于一个文件集或一个语料库中其中一个文件的重要程度
Tf- term frequency
idf- 逆向文档频率：inverse document frequency: 总文件数目除以包含该词语之文件的数目，在将结果取10的对数
"""


def tfidf_demo():
    text = ["life is short, i like python",
            "life is too long, i dislike python"]
    transfer = TfidfVectorizer()
    data_vector = transfer.fit_transform(text)
    print("feature:\n", transfer.get_feature_names(), "\n")
    print(text, "\n")
    print("data:\n", data_vector.toarray())


"""归一化/无量纲化"""


def min_max():
    data = pd.read_csv('data/dating.csv')
    data = data.iloc[:, :3]
    print("data pre\n", data)
    transfer = MinMaxScaler(feature_range=[0, 1])
    data_new = transfer.fit_transform(data)
    print("after\n", data_new)
    return None


"""标准化"""


def standerlize():
    data = pd.read_csv('data/dating.csv')
    data = data.iloc[:, :3]
    print("data pre\n", data)
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print("after\n", data_new)
    return None

# low variance filter
def variance_demo():
    data=pd.read_csv("data/factor_returns.csv")
    data=data.iloc[:,1:-2]
    print("data\n", data)
    transfer=VarianceThreshold()
    data_new=transfer.fit_transform(data)
    print("data_new\n", data_new.shape)
    # cal two vaaiable depency
    r1=pearsonr(data.pe_ratio,data.pb_ratio)
    print("xiangguanxishu:\n", r1)
    r2=pearsonr(data.revenue,data.total_expense)
    print("revenue total_expense xiangguanxishu :\n", r2)
    plt.figure(figsize=(8,8),dpi=100)
    plt.scatter(data.revenue,data.total_expense)
    plt.show()
    return None

def pca_demo():
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    data_new=transfer=PCA(n_components=0.95).fit_transform(data)
    print("data\n",data_new)


def day01_instacart():
    order_products = pd.read_csv("data/instacart/order_products__prior.csv")
    products = pd.read_csv("data/instacart/products.csv")
    orders = pd.read_csv("data/instacart/orders.csv")
    aisles = pd.read_csv("data/instacart/aisles.csv")
    print("aisles\n", aisles)
    tab1=pd.merge(aisles,products,on=["aisle_id","aisle_id"])
    print("tab1\n",tab1)
    tab2=pd.merge(tab1, order_products, on=["product_id", "product_id"])
    print("tab2\n", tab2)
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])
    print("tab3\n", tab3.head())
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])
    print(table.shape)
    data = table
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    print("data_new shape: ",data_new.shape)
    return



if __name__ == "__main__":
    # dict_demo()
    # print('\n')
    # min_max()
    tfidf_demo()
    # dict_demo()
