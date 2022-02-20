import keras.datasets.imdb
import tensorflow as tf


def TFconcat():
    # the condition which tensor can be concated is only one axis different, rest are same
    """
    从语法上来说，拼接合并操作可以在任意的维度上进行，唯一的约束是非合并维度的长度必须一致
    拼接操作直接在现有维度上合并数据，并不会创建新的维度
    """
    a = tf.random.normal([4, 35, 8])  # 模拟成绩册 A
    b = tf.random.normal([6, 35, 8])  # 模拟成绩册 B
    c = tf.concat([a, b], axis=0)  # 拼接合并成绩册
    print(a.shape, "\n", "*" * 100, "\n", b.shape, "\n", "*" * 100, "\n", c.shape, "\n", "*" * 100, "\n")
    a = tf.random.normal([10, 35, 4])
    b = tf.random.normal([10, 60, 4])
    c = tf.concat([a, b], axis=1)
    print(a.shape, "\n", "*" * 100, "\n", b.shape, "\n", "*" * 100, "\n", c.shape, "\n", "*" * 100, "\n")


def TFStack():
    """
    堆叠 如果在合并数据时，希望创建一个新的维度，则需要使用 tf.stack 操作。
    tf.stack 也需要满足张量堆叠合并条件，它需要所有待合并的张量 shape 完全一致才可合并。
    """
    a = tf.random.normal([35, 8], dtype=tf.float64)
    b = tf.random.normal([35, 8], dtype=tf.float64)
    c = tf.stack([a, b], axis=0)  # axis --> insert position
    print(a.shape, "\n", "*" * 100, "\n", b.shape, "\n", "*" * 100, "\n", c.shape, "\n", "*" * 100, "\n")


def TFSplit():
    x = tf.random.normal([10, 35, 8])
    # 等长切割为 10 份
    """
    1. 如果num_or_size_splits 传入的 是一个整数，那直接在axis=D这个维度上把张量平均切分成几个小张量
    2. 如果num_or_size_splits 传入的是一个向量（这里向量各个元素的和要跟原本这个维度的数值相等）就根据这个向量有几个元素分为几项）
    """
    result = tf.split(x, num_or_size_splits=[4, 2, 2, 2], axis=0)
    print(len(result))  # 返回的列表为 10 个张量的列表
    print(result[0])


def TFmax():
    x = tf.random.normal([4, 10])
    print(x)
    print('.' * 100)
    print(tf.reduce_mean(x, axis=0))


def tensorCompare():
    out = tf.random.normal([100, 10])  # 10 class
    print(out)
    out = tf.nn.softmax(out, axis=1)
    print("." * 100)
    print(out)
    pred = tf.argmax(out, axis=1)  # dimension=0 按列找, dimension=1 按行找
    print(pred)
    y = tf.random.uniform([100], dtype=tf.int64, maxval=10)
    out = tf.equal(pred, y)
    print(out)
    out = tf.cast(out, dtype=tf.int32)
    correct = tf.reduce_sum(out)

    print(correct)
    print("accu: ", correct / 100)


def TFPadding():
    a = tf.constant([1, 2, 3, 4, 5, 6])
    b = tf.constant([7, 8, 1, 6])
    b = tf.pad(b, [[0, 2]])
    print(b)
    c = tf.stack([a, b])
    print(c)


def TFPaddingExample():
    print("Sentence Padding")
    total_words = 10000
    max_review_len = 80  # 最大句子长度
    embedding_len = 100
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    print(x_train.shape, x_test.shape)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len, truncating="post",
                                                         padding="post")
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len, truncating="post",
                                                        padding="post")
    print(x_train.shape, x_test.shape)
    print("Image Padding")
    a = tf.random.normal([4, 28, 28, 3])
    b = tf.pad(a, [[0, 0], [2, 2], [2, 2], [0, 1]])  # 第一个[2,2]表示给待补充的TENSOR上下各添加2行，第二个[2，2] 表示给待补充的Tensor左右各添加2行
    print(b)


def TFGather():
    x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int32)  # 成绩册张量
    print(tf.gather(x,[0,1],axis=0) )# 在班级维度收集第 1~2 号班级成绩册)

if __name__ == "__main__":
    # TFconcat()
    # TFStack()
    # TFSplit()
    # TFmax()
    # tensorCompare()
    # TFPadding()
    # TFPaddingExample()
    TFGather()