#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import platform
sys = platform.system()
if sys == "Windows":
    print("#"*20 + '当前系统为Windows系统' + "#"*20)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu。
import json
from PIL import Image
from skimage import io, transform  # skimage模块下的io transform(图像的形变与缩放)模块
import glob  # glob 文件通配符模块
import os  # os 处理文件和目录的模块
import tensorflow as tf
import numpy as np  # 多维数据处理模块
import time
from Setting import BASEPATH
import warnings


warnings.filterwarnings("ignore")
def train(n=16):
    if os.path.exists(BASEPATH + 'ok.txt'):
        os.remove(BASEPATH + 'ok.txt')
    # 数据集地址
    path = BASEPATH + 'train_photos/'
    # 模型保存地址
    model_path = BASEPATH + '/model/fc_model.ckpt'

    # 将所有的图片resize成100*100
    w = 100
    h = 100
    c = 3


    # 读取图片+数据处理
    def read_img(path):
        # os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
        # os.path.isdir(path)判断path是否是目录
        # b = [x+x for x in list1 if x+x<15 ]  列表生成式,循环list1，当if为真时，将x+x加入列表b
        cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        imgs = []
        labels = []
        save_TitleName = {}
        for idx, folder in enumerate(cate):
            # glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表
            name = folder.split('/')[-1]
            save_TitleName[str(idx)] = name
            for im in glob.glob(folder + '/*.jpg'):
                # 输出读取的图片的名称
                print('reading the images:%s' % (im))
                # io.imread(im)读取单张RGB图片 skimage.io.imread(fname,as_grey=True)读取单张灰度图片
                # 读取的图片
                img = Image.open(im)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(im)
                L_image = Image.open(im)
                L_image = L_image.convert('L')
                img = np.array(L_image)
                whiteindex = (img > 200)
                img[whiteindex] = 255
                blackindex = (img <= 200)
                img[blackindex] = 0
                out = Image.fromarray(img.astype('uint8')).convert('RGB')
                img = np.array(out)
                # skimage.transform.resize(image, output_shape)改变图片的尺寸
                # img = transform.resize(img, (w, h))
                # 将读取的图片数据加载到imgs[]列表中
                # imgs.append(img)
                # 将图片的label加载到labels[]中，与上方的imgs索引对应
                # labels.append(idx)
                # 筛选掉不符合要求的图像
                try:
                    if img.shape[2] == 3:
                        img = transform.resize(img, (w, h))
                        imgs.append(img)
                        labels.append(idx)
                    else:
                        print('---->>>> %s ：不是100*100*3的图片！请注意！！！' % (im))
                except:
                    print("报错了！！！")
                    continue
        titleNamePath = BASEPATH + 'titleNameDict.txt'
        with open(titleNamePath, 'w') as f:
            f.write(json.dumps(save_TitleName))
        # 将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
        return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


    # 调用读取图片的函数，得到图片和labels的数据集
    data, label = read_img(path)

    # 打乱顺序
    # 读取data矩阵的第一维数（图片的个数）
    num_example = data.shape[0]
    # 产生一个num_example范围，步长为1的序列
    arr = np.arange(num_example)
    # 调用函数，打乱顺序
    np.random.shuffle(arr)
    # 按照打乱的顺序，重新排序
    data = data[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    # -----------------构建网络----------------------
    # 本程序cnn网络模型，共有7层，前三层为卷积层，后三层为全连接层，前三层中，每层包含卷积、激活、池化层
    # 占位符设置输入参数的大小和格式
    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


    def inference(input_tensor, train, regularizer):
        # -----------------------第一层----------------------------
        with tf.variable_scope('layer1-conv1'):
            # 初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
            conv1_weights = tf.get_variable("weight", [5, 5, 3, 32],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            # 初始化偏置conv1_biases，数量为32个
            conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
            # 卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据，
            # conv1_weights为权重，strides=[1, 1, 1, 1]表示左右上下滑动步长为1，padding='SAME'表示输入和输出大小一样，即补0
            conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            # 激励计算，调用tensorflow的relu函数
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        with tf.name_scope("layer2-pool1"):
            # 池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        # -----------------------第二层----------------------------
        with tf.variable_scope("layer3-conv2"):
            # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
            conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # -----------------------第三层----------------------------
        # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        with tf.variable_scope("layer5-conv3"):
            conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

        with tf.name_scope("layer6-pool3"):
            pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # -----------------------第四层----------------------------
        # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        with tf.variable_scope("layer7-conv4"):
            conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

        with tf.name_scope("layer8-pool4"):
            pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            nodes = 6 * 6 * 128
            reshaped = tf.reshape(pool4, [-1, nodes])
            # 使用变形函数转化结构
        # -----------------------第五层---------------------------
        with tf.variable_scope('layer9-fc1'):
            # 初始化全连接层的参数，隐含节点为1024个
            fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))  # 正则化矩阵
            fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
            # 使用relu函数作为激活函数
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            # 采用dropout层，减少过拟合和欠拟合的程度，保存模型最好的预测效率
            if train: fc1 = tf.nn.dropout(fc1, 0.5)
        # -----------------------第六层----------------------------
        with tf.variable_scope('layer10-fc2'):
            # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
            fc2_weights = tf.get_variable("weight", [1024, 512],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
            if train: fc2 = tf.nn.dropout(fc2, 0.5)
        # -----------------------第七层----------------------------
        with tf.variable_scope('layer11-fc3'):
            # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
            fc3_weights = tf.get_variable("weight", [512, n],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
            fc3_biases = tf.get_variable("bias", [n], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc2, fc3_weights) + fc3_biases  # matmul矩阵相乘
        # 返回最后的计算结果
        return logit


    # ---------------------------网络结束---------------------------
    # 设置正则化参数为0.0001
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    # 将上述构建网络结构引入
    logits = inference(x, False, regularizer)

    # (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
    b = tf.constant(value=1, dtype=tf.float32)
    logits_eval = tf.multiply(logits, b, name='logits_eval')  # b为1

    # 设置损失函数，作为模型训练优化的参考标准，loss越小，模型越优
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    # 设置整体学习率为α为0.001
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # 设置预测精度
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 定义一个函数，按批次取数据
    def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


    # 训练和测试数据，可将n_epoch设置更大一些


    # 迭代次数
    n_epoch = 16
    # 每次迭代输入的图片数据
    batch_size = 64
    saver = tf.train.Saver(max_to_keep=1)  # 可以指定保存的模型个数，利用max_to_keep=4，则最终会保存4个模型（
    with tf.Session() as sess:
        # 初始化全局参数
        sess.run(tf.global_variables_initializer())
        # 开始迭代训练，调用的都是前面设置好的函数或变量
        for epoch in range(n_epoch):
            print(" >>> --------第 %s 训练开始-------- <<<" % (epoch + 1))
            start_time = time.time()

            # training#训练集
            train_loss, train_acc, n_batch = 0, 0, 0
            for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
                _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
                train_loss += err;
                train_acc += ac;
                n_batch += 1
                print("   train loss: %f" % (np.sum(train_loss) / n_batch))
                print("   train acc: %f" % (np.sum(train_acc) / n_batch))

            # validation#验证集
            val_loss, val_acc, n_batch = 0, 0, 0
            for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
                err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
                val_loss += err;
                val_acc += ac;
                n_batch += 1
                print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
                print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
            # 保存模型及模型参数
            if epoch % 2 == 0:
                # saver.save(sess, model_path, global_step=epoch)
                saver.save(sess, model_path)
    with open(BASEPATH + 'ok.txt', 'w')as f:
        f.write('')

if __name__ == '__main__':
    n = 0
    for i in os.listdir(BASEPATH + 'train_photos/'):
        if os.path.isdir(BASEPATH + 'train_photos/' + i):
            n += 1
    train(n)

