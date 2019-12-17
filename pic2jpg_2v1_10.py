#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import re
import time

from Setting import BASEPATH
import numpy as np
from PIL import Image

def getDir(path):
    # print(path)
    # print(os.listdir(path))
    return [path + x for x in os.listdir(path) if os.path.isdir(path + x)]

def getFiles(path):
    return [path + x for x in os.listdir(path) if os.path.isfile(path + x)]

def pic2jpg(picpath):
    img = Image.open(picpath)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    h_old = img.shape[0]
    w_old = img.shape[1]
    print(img.shape)
    h_new = h_old
    w_new = w_old
    if h_old * 2 < w_old:
        h_new = int(w_old / 2)
    if h_old * 2 > w_old:
        w_new = int(h_old * 2)
    print(h_new, w_new)
    sizeScale = random.randint(100,150) / 100
    w_new = int(w_new * sizeScale)
    h_new = int(h_new * sizeScale)
    w_l = random.randint(0, (w_new - w_old))
    w_r = w_new - w_old - w_l
    h_u = random.randint(0, (h_new - h_old))
    h_d = h_new - h_old - h_u

    matrix_pad = np.pad(img, pad_width=((h_u, h_d),  # 向上填充1个维度，向下填充两个维度
                                        (w_l, w_r),  # 向左填充2个维度，向右填充一个维度
                                        (0, 0))  # 通道数不填充
                        , mode="constant",  # 填充模式
                        constant_values=(255, 255))
    im = Image.fromarray(matrix_pad)
    filename = re.search(r'./originalImage/(.*)', picpath, re.S).group(1)
    savepath = BASEPATH + 'train_photos/' +filename
    dirpath, tempfilename = os.path.split(savepath)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    savepath = dirpath + '/' + str(int(time.time()*1000)) + str(random.randint(0, 100)) + '.jpg'
    print(savepath)
    im.save(savepath)

if __name__ == '__main__':
    path = './originalImage/'
    cate = getDir(path)
    for i in cate:
        print(i)
        fileList = getFiles(i+'/')
        for i in fileList:
            print(fileList)
            for _ in range(80):
                try:
                    pic2jpg(i)
                except:
                    pass



