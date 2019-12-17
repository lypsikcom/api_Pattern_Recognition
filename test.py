#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import zipfile

import requests
import sys

# sys.path.append(".")
# url = 'http://localhost:5000/recognition'
# # url = 'http://localhost:5000/upload'
# # url = 'https://httpbin.org/post'
#
# files = {'file': open('111.jpg', 'rb')}
# r = requests.post(url, files=files)
# print(r.text)

# zf = zipfile.ZipFile('./Data/Recognition/train.zip')
# zf.extractall(path='./Data/Recognition/train_photos/')
# zf.close()
# print('adfsfsd')
# n = 0
# for i in os.listdir('./Data/Recognition/train_photos/'):
#     print(i)
#     if os.path.isdir('./Data/Recognition/train_photos/' + i):
#         n += 1
# print("n=%s" % (str(n)))
# from skimage import io
# #
# picpath = '154.jpg'
# img = io.imread(picpath)
# # print(img.shap[0])
# # print(img.shap[1])
# # print(img.shap[2])
# # print(img)
# # print('='*120)
# # print(img[:][:][:])
# print(type(img))
#
# # if img.shap[2] == 4:
# # # if img.shape[2] == 3:
# #     print('1')
#
#
# from PIL import Image
# img=Image.open('154.jpg')
# print(img.mode)
# img=img.convert('RGB')
# print(img.mode=='RGB')
import requests

data = {
    'url':'http://htsc.dgyt.petrochina/public/0/images/95/7658d73634c76c4ab9c8f1f8a587493e-196_61ht1107wd158.jpg',
}
# url = 'http://localhost:5000/recognition'
url = 'http://10.76.31.105:9002/recognition'
res = requests.post(url,data=data)

print(res.text)

