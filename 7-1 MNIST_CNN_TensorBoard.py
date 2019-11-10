
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每批次的大小
batch_size = 100
# 计算一共需要多少批次
n_batch = mnist.train.num_examples // batch_size

test_n_batch = mnist.test.num_examples // batch_size

# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# 初始化权重
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# 卷积层
def conv2d(x, W):
    # x的输入shape: [batch, in_height, in_weight, in_channels]
    # W(卷积核)的shape: [filter_height, filter_width, in_channels, out_channels]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 池化层
def max_pool_2x2(x):
    # ksize [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 命名空间
with tf.name_scope('input'):
    # 定义placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10],  name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值核权重
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5,5,1,32], name='W_conv1')  #5x5的卷积核（采样窗口），32个卷积核从一个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每个卷积核一个偏置值

    # 把x_image和权重向量进行卷积，在加上偏置值，然后使用relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 池化

with tf.name_scope('Conv2'):
    # 初始化第一个卷积层的权值核权重
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5x5的卷积核（采样窗口），64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每个卷积核一个偏置值

    # 把x_image和权重向量进行卷积，在加上偏置值，然后使用relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 池化

'''        
with tf.name_scope('Conv3'):
    # 初始化第一个卷积层的权值核权重
    with tf.name_scope('W_conv3'):
        W_conv3 = weight_variable([5, 5, 64, 64], name='W_conv3')  # 5x5的卷积核（采样窗口），64个卷积核从32个平面抽取特征
    with tf.name_scope('b_conv3'):
        b_conv3 = bias_variable([64], name='b_conv3')  # 每个卷积核一个偏置值

    # 把x_image和权重向量进行卷积，在加上偏置值，然后使用relu激活函数
    with tf.name_scope('conv2d_3'):
        conv2d_3 = conv2d(h_pool2, W_conv3) + b_conv3
    with tf.name_scope('relu'):
        h_conv3 = tf.nn.relu(conv2d_3)
    with tf.name_scope('h_pool3'):
        h_pool3 = max_poll_2x2(h_conv3)  # 池化
'''
# 28*28的图片经过第一次卷积后还是28*28， 第一次池化后变为14*14
# 第二次卷积后为14*14， 第二次池化变为7*7
# 经过上面的操作后得到64张的7*7的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1') # 上一层有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1') # 1024

    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')
    # 第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # 使用keep_prob
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接权重
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = weight_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction), name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1)) # argmax返回一维向量中最大值所在的位置
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    # 训练
    for i in range(1000):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.5})
            # 记录训练集计算的参数
            summary = sess.run(merged, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.5})
            train_writer.add_summary(summary, i)

        # 记录测试集计算的参数
        for batch in range(test_n_batch):
            batch_xs, batch_ys = mnist.test.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            # 记录训练集计算的参数
            summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            test_writer.add_summary(summary, i)
        #if 1%100 == 0:
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:0.5})
        test_acc  = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:0.5})
        print("Iter" + str(i) + ", Train Accuracy: " + str(train_acc) + ", Testing Accuracy: " + str(test_acc))
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(100):
        # 训练模型
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})  # keep_prob:1.0 所有元素都保留
        train_writer.add_summary(summary, i)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 1 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))
