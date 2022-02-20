import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets  # 导入经典数据集加载模块


def DataSet():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:',
          y_test)
    """数据计内存中，需要将其转换为dataset"""
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    # 随机打散--》避免模型尝试记住标签信息
    train_db = train_db.shuffle(10000)
    # 批训练
    train_db = train_db.batch(128)
    # 预处理
    train_db = train_db.map(preprocess)
    return train_db


def preprocess(x, y):  # 自定义的预处理函数
    # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    # 打平
    # 转成整型张量
    # one-hot 编码
    # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能
    return x, y


def network():
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    # 第二层的参数
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    # 第三层的参数
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3


def train_epoch(train_dataset, w1, b1, w2, b2, w3, b3, epoch, lr=0.001):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # 第一层计算， [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b,256] + [b, 256]
            h1 = x @ w1 + tf.broadcast_to(b1, (x.shape[0], 256))
            h1 = tf.nn.relu(h1)  # 通过激活函数

            # 第二层计算， [b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # 输出层计算， [b, 128] => [b, 10]
            out = h2 @ w3 + b3

            # 计算网络输出与标签之间的均方差， mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y - out)
            # 误差标量， mean: scalar
            loss = tf.reduce_mean(loss)

            # 自动梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # 梯度更新， assign_sub 将当前值减去参数值，原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        print(epoch, step, 'loss:', loss.numpy())

    return loss.numpy()


if __name__ == "__main__":
    datasets = DataSet()
    w1, b1, w2, b2, w3, b3 = network()
    iterations = 20
    for i in range(20):
        loss=train_epoch(datasets,w1, b1, w2, b2, w3, b3,i)
