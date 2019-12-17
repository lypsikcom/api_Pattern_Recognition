import os
import platform
import time

sys = platform.system()
if sys == "Windows":
    print("#"*20 + '当前系统为Windows系统' + "#"*20)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 这一行注释掉就是使用cpu，不注释就是使用gpu。
import json
from PIL import Image
from skimage import io, transform
import tensorflow as tf
import numpy as np
import glob  # glob 文件通配符模块
from Setting import BASEPATH

# 此程序作用于进行简单的预测，取5个图片来进行预测，如果有多数据预测，按照cnn.py中，读取数据的方式即可

def testPic(picpath=''):
    start = time.time()
    path = BASEPATH + 'test_photos/'
    # 类别代表字典
    # flower_dict = {0: 'dasiy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    # flower_dict = {0: 'cat', 1: 'lty', 2: 'lyp', 3: 'x2'}
    flower_dict = {}
    str_dict = ''
    with open((BASEPATH + "titleNameDict.txt"), 'r') as f:
        str_dict = f.read()
    flower_dict = json.loads(str_dict)
    # print(flower_dict)

    w = 100
    h = 100
    c = 3

    def pic2Standard(picpath):
        img = Image.open(picpath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        h_old = img.shape[0]
        w_old = img.shape[1]
        #print(img.shape)
        h_new = h_old
        w_new = w_old
        if h_old * 2 < w_old:
            h_new = int(w_old / 2)
        if h_old * 2 > w_old:
            w_new = int(h_old * 2)
        #print(h_new,w_new)
        h_u = int((h_new - h_old) / 2)
        h_d = h_new - h_old - h_u
        w_l = int((w_new - w_old) / 2)
        w_r = w_new - w_old - w_l
        #print(h_u,h_d,w_l,w_r)
        matrix_pad = np.pad(img, pad_width=((h_u, h_d),  # 向上填充u个维度，向下填充d个维度
                                            (w_l, w_r),  # 向左填充l个维度，向右填充r个维度
                                            (0, 0))      # 通道数不填充
                            , mode="constant",  # 填充模式
                            constant_values=(255, 255))
        img = Image.fromarray(matrix_pad)
        img.save(picpath)
        return

    # 读取图片+数据处理
    def read_img(path):
        # os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
        # os.path.isdir(path)判断path是否是目录
        # b = [x+x for x in list1 if x+x<15 ]  列表生成式,循环list1，当if为真时，将x+x加入列表b
        cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        imgs = []

        for idx, folder in enumerate(cate):
            # glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表
            for im in glob.glob(folder + '/*.jpg'):
                # 输出读取的图片的名称
                #print('reading the images:%s' % (im))
                # io.imread(im)读取单张RGB图片 skimage.io.imread(fname,as_grey=True)读取单张灰度图片
                # 读取的图片
                img = Image.open(im)
                pic2Standard(im)
                img = io.imread(im)
                # skimage.transform.resize(image, output_shape)改变图片的尺寸
                img = transform.resize(img, (w, h))
                # 将读取的图片数据加载到imgs[]列表中
                imgs.append(img)
                # 将图片的label加载到labels[]中，与上方的imgs索引对应
            # labels.append(idx)
        # 将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
        return np.asarray(imgs, np.float32)

    def read_one(picpath):
        imgs = []
        pic2Standard(picpath)
        img = io.imread(picpath)
        img = transform.resize(img, (w, h))
        imgs.append(img)
        return np.asarray(imgs, np.float32)

    # 调用读取图片的函数，得到图片和labels的数据集
    if not picpath:
        data = read_img(path)
    else:
        data = read_one(picpath)
    dict_response = []
    with tf.Session() as sess:
        # modelPath = ''
        # for i in glob.glob(BASEPATH + 'model/' + '/*.meta'):
        #     if not modelPath:
        #         modelPath = i
        #     else:
        #         if os.path.getmtime(modelPath) < os.path.getmtime(i):
        #             modelPath = i
        saver = tf.train.import_meta_graph(BASEPATH + 'model/fc_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(BASEPATH + 'model/'))
        # sess：表示当前会话，之前保存的结果将被加载入这个会话
        # 设置每次预测的个数
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")  # eval功能等同于sess(run)

        classification_result = sess.run(logits, feed_dict)

        # 打印出预测矩阵
        # print(classification_result)
        # 打印出预测矩阵每一行最大值的索引
        # print(tf.argmax(classification_result, 1).eval())
        # 根据索引通过字典对应花的分
        output = []
        # output = tf.argmax(classification_result, 1).eval()
        # for i in range(len(output)):
        #     print("第", i + 1, "朵花预测:" + flower_dict[output[i]])
        # print(classification_result)
        output = tf.math.top_k(classification_result,3)
        output = output.indices.eval()
        # print(output)
        # print(output.values[0][0].eval())
        # print(output.indices[0][0].eval())
        # print(flower_dict)
        for i in range(len(output)):
            # print("第", i + 1, "朵花预测: 1: " + flower_dict[str(output[i][0])] + ',2: ' + flower_dict[str(output[i][1])] + ',3: ' + flower_dict[str(output[i][2])])
            dict_response.append({})
            dict_response[-1]['0'] = flower_dict[str(output[i][0])]
            dict_response[-1]['1'] = flower_dict[str(output[i][1])]
            dict_response[-1]['2'] = flower_dict[str(output[i][2])]
    end = time.time()
    print("识别图片用时：%s (S)" % str(end - start))
    return dict_response

if __name__ == '__main__':
    print(testPic('./Data/Recognition/test_photos/test/微信图片_20191125192715.jpg'))
    # print(testPic())

