import keras.layers
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def basic_shuZiLeiXing():
    print("tensorflow constant")
    aa = tf.constant(1.2, dtype=tf.float64)
    print(aa)
    print(type(aa), tf.is_tensor(aa))
    print("tensorflow vector")
    aa = tf.constant([1, 2])
    print(aa)
    print(type(aa), tf.is_tensor(aa))
    print("tensorflow matrix")
    aa = tf.constant([[1, 2], [1.2, 3.5]])
    print(aa)
    print(type(aa), tf.is_tensor(aa))
    print(aa.numpy().size)


def basic_String():
    print("create a string tensor")
    a = tf.constant("Hello Deep Learning")
    print(a)
    print("string tensor length")
    print(tf.strings.length(a))
    print("string char to lower")
    print(tf.strings.lower(a))
    print("string char to upper")
    print(tf.strings.upper(a))
    print("split the string")
    aa = tf.strings.split(a, " ")
    print(aa)
    print("join two tensor")
    print(tf.strings.join([a, aa], separator=" "))


def gradientVarible():
    a = tf.constant([[[1.2, 2.3], [4, 5], [4, 4]], [[2, 6], [4, 5], [8, 9]]])
    aa = tf.Variable(a)
    print(aa.shape)


def create_tensor():
    a = tf.convert_to_tensor([1.2, 2], dtype=tf.float64)
    print(a)
    a = tf.convert_to_tensor(np.arange(16).reshape(2, 2, 4), dtype=tf.int64)
    print(a)
    b = tf.constant([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]])
    print(b)


def custom_tensor():
    a = tf.fill((2, 2), 100)
    print(a)


def tensorWithNormalDistribution():
    print("Normal Distribution")
    a = tf.random.normal((2, 2), mean=2, stddev=5)
    print(a)
    print("uniform Distribution")
    a = tf.random.uniform((2, 2), minval=20, maxval=100, dtype=tf.int64)
    print(a)


def tensorWithXuLie():
    a = tf.range(0, 10, delta=1, dtype=tf.double)
    print(a)


def ConstantTest():
    out = tf.random.uniform([4, 10])  # 随机模拟网络输出
    print(out)
    y = tf.constant([2, 3, 2, 0])  # 随机构造样本真实标签
    y = tf.one_hot(y, depth=10)  # one-hot 编码
    print("\n", y)
    loss = tf.keras.losses.mse(y, out)  # 计算每个样本的 MSE
    loss = tf.reduce_mean(loss)  # 平均 MSE,loss 应是标量
    print(loss)


def VectorTest():
    z = tf.random.normal([4, 2])
    b = tf.zeros([2])  # 创建偏置向量
    z = z + b  # 累加上偏置向量
    fc = keras.layers.Dense(3)  # 创建一层 Wx+b，输出节点为 3
    # 通过 build 函数创建 W,b 张量，输入节点为 4
    fc.build(input_shape=(2, 4))
    print(fc.bias)


def MatrixTest():
    x = tf.random.normal((2, 4))
    w = tf.ones([4, 3])
    b = tf.zeros((3))
    o = x @ w + b
    print(o)
    fc = tf.keras.layers.Dense(3)  # 定义全连接层的输出节点为 3
    fc.build(input_shape=(2, 4))  # 定义全连接层的输入节点为 4
    out = fc(x, )
    print(out)  # 查看权值矩阵 W


def TensorTest():
    x = tf.random.normal([4, 32, 32, 3])
    layers = keras.layers.Conv2D(16, kernel_size=(3, 3))
    out = layers(x)
    print(out.shape)


def indexTensor():
    x = tf.random.normal([4, 3, 32, 32])
    print(x[0][1][2])


def GenerateTensor():
    x = tf.range(96)
    x = tf.reshape(x, [2, 3, 4, 4])  # 2--> batch size, 3 --> channel size, 4--> height, 4--> width
    print(x, "\n", x.shape, x.ndim)
    x = tf.reshape(x, [2, -1])
    print(x, "\n", x.shape, x.ndim)
    x = tf.reshape(x, [2, 4, 12])  # 2--> batch size,  4--> height, 12--> width
    print(x, "\n", x.shape, x.ndim)
    x = tf.reshape(x, [2, 16, 3])  # 2--> batch size,  16--> height, 3--> width
    print(x, "\n", x.shape, x.ndim)


def changeDim():
    x = tf.random.uniform([28, 28], maxval=10, dtype=tf.int64)
    print(x, "\n", x.shape, x.ndim)
    x = tf.expand_dims(x, axis=2)  # insert the new dim after the width -->[b,h,w]-->[28,28,1]
    print(x[27], "\n", x.shape, x.ndim)
    x = tf.expand_dims(x, axis=0)  # inset the new dim before the height --> [b,c,h,w]-->[1,28,28,1]
    print(x[0], "\n", x.shape, x.ndim)
    print("." * 100)
    print("删除维度")  # delete the dim can not remove the pre-setted value
    x = tf.squeeze(x, axis=0)
    x = tf.squeeze(x, axis=2)
    print(x, "\n", x.shape, x.ndim)


def tensorTransDim():
    x = tf.random.normal([1, 32, 32, 3])
    print(x)
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    print(x)


if __name__ == "__main__":
    # basic_shuZiLeiXing()
    # basic_String()
    # gradientVarible()
    create_tensor()
    # custom_tensor()
    # tensorWithNormalDistribution()
    # tensorWithXuLie()
    # ConstantTest()
    # MatrixTest()
    # TensorTest()
    # indexTensor()
    # GenerateTensor()
    # changeDim()
    # tensorTransDim()
