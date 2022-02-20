from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
import warnings
from numpy import bool8, int16, random


def numpy1():
    a = np.arange(15).reshape(3, 5)
    # ndim-->数据维度
    # shape array dimension
    # size-->item number
    # dtpye-->data type
    a = np.array(a)
    print(a.size)
    print(a.shape)
    print(a.ndim)
    print(a.dtype)


def numpy2():
    a = np.array([2, 3, 4])
    b = np.array([(1.5, 2, 3), (4, 5, 6)])
    c = np.zeros((5, 5), dtype=np.int16)
    d = np.ones((5,), dtype=np.float64)

    # ndim-->数据维度
    # shape array dimension
    # size-->item number
    # dtpye-->data type
    print(a.size)
    print(a.shape)
    print(a.ndim)
    print(b.dtype)
    print(b)
    print(c)
    print(d.ndim)


def numpy3():
    a = np.arange(16).reshape(2, 2, 2, 2)
    b = np.arange(16).reshape(2, 2, 4)
    c = np.arange(16).reshape(2, 8)
    d = np.arange(16)
    # ndim-->数据维度
    # shape array dimension
    # size-->item number
    # dtpye-->data type
    print("a.size: ", a.size)
    print("a.shape: ", a.shape)
    print("a.ndim: ", a.ndim)
    print("a: ", a)
    print("b.size: ", b.size)
    print("b.shape: ", b.shape)
    print("b.ndim: ", b.ndim)
    print("b: ", b)
    print("c.size: ", c.size)
    print("c.shape: ", c.shape)
    print("c.ndim: ", c.ndim)
    print("c: ", c)


def numpy4():
    a = np.array([20, 30, 40, 50])
    b = np.arange(4)
    c = a - b
    print(c)
    # **-- 平方
    print(b ** 2)
    # @-- matrix production
    print(b @ a)

    rg = np.random.default_rng(1)
    a = np.ones((2, 3), dtype=int)
    b = rg.random((2, 3))
    a *= 3
    print(a)
    b += a
    print(b)
    a = rg.random((2, 3))
    print("\n", a)
    print(" min value {}".format(a.min()))
    print(" each row min value{}".format(a.min(axis=1)))  # 竖直方向上的每一行的最小值
    print("sum value {}".format(a.sum()))  # sum 整个npArray
    print("row cumsum {}".format(a.sum(axis=1)))  # sum 每一行的值
    print("each col sum {}".format(a.sum(axis=0)))  # sum 每一列的值


def numpy5():
    t1 = np.array([random.random() for i in range(10)])
    print(t1)
    print(t1.dtype)
    t2 = t1.round(0)
    t3 = t2.astype(np.bool8)  # 数据类型转换器
    print(t3)


def numpy6():
    arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
    print(arr)
    arr = np.sort(arr)
    print(arr)


def numpy7():
    t2 = np.arange(100000).reshape(20, 5000)
    print(t2)
    print("-" * 100)
    print("取行")
    print(t2[2])
    print("连续多行")
    print(t2[2:, ])
    print("取不连续的多行")
    print(t2[[2, 8, 10],])
    print("取单列")
    print(t2[:, 3])
    print("连续取列")
    print(t2[:, [3, 9, 5]])
    print("除第一行以外的列")
    print(t2[1:, ])
    print("除第一行和第一列")
    print(t2[1:, 1:])
    print("第19行第590的数据")
    print(t2[19, 590])
    print("dtype", t2[19, 590].dtype)


def numpy8():
    print("ndarray merge:")
    t2 = np.arange(100000).reshape(20, 5000).astype(np.float64)
    array = [random.randint(100000, 200001) for i in range(100000)]
    t3 = np.array(array).reshape(20, 5000)
    t4 = np.vstack((t2, t3))
    print(t2.shape)
    print(t3.shape)
    print(t4.shape)


def numpy9():
    print("numpy array NaN 操作")
    t5 = np.array([np.random.randint(0, 10) for i in range(10)]).reshape(2, 5)
    print("orginal ndarray is\n", t5)
    print("ndarray transpose: \n", t5.T)
    print("ndarray check how many nonzero item:")
    print(np.count_nonzero(t5 != t5))
    print("ndarray check if contains the NaN:")
    print(np.isnan(t5))
    print("assgin the NaN to the ndarray:")
    t5 = t5.astype(np.float64)
    t5[0, [[1, 3, 2]]] = np.nan
    print(t5, "\n")
    print("remove the NaN with the mean value")
    # nan colum means
    col = np.nanmean(t5, axis=0)
    print("The col mean value which contains the NaN value", col, "\n")
    # get nan index
    index = np.where(np.isnan(t5))  # frist array is the row index, the second array is the col index
    print("NaN value index:", index)

    t5[index] = np.take(col, index[1])
    print("After Replacing:\n", t5)


if __name__ == "__main__":
    print("numpy 基本数据结构：")
    numpy1()
    print("*" * 100)
    numpy2()
    print("*" * 100)
    numpy3()
    print("*" * 100)
    print("numpy 基本操作：")
    numpy4()
    print("*" * 100)
    numpy5()
    print("*" * 100)
    numpy6()
    print("*" * 100)
    print("numpy 结构操作")
    numpy7()
    print("*" * 100)
    numpy8()
    print("*" * 100)
    numpy9()
    print("*" * 100)
