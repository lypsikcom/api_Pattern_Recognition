import ctypes
import inspect
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pymysql

from Setting import BASEPATH
from train_model import train
from app import deleteFiles
import warnings


warnings.filterwarnings("ignore")
def checkMission():
    # 开创线程池
    while 1:
        # 检查超时线程 ，并关闭
        # 读取需要执行的程序，放入线程池，每次取未执行或者报错的 创建时间小于1小时 growby 上次处理时间最小的
        # future = executor.submit(test_function, 4, 8)
        try:
            if not os.path.exists(BASEPATH + 'train_photos/'):
                os.makedirs(BASEPATH + 'train_photos/')
                os.makedirs(BASEPATH + 'model/')
                os.makedirs(BASEPATH + 'test_photos/test')
            startpath = BASEPATH + 'start.txt'
            if os.path.exists(startpath):
                os.remove(startpath)
                deleteFiles(BASEPATH + 'train_photos/')
                zippath = BASEPATH + 'train.zip'
                zf = zipfile.ZipFile(zippath)
                zf.extractall(path=BASEPATH + 'train_photos/')
                zf.close()
                n = 0
                for i in os.listdir(BASEPATH + 'train_photos/'):
                    if os.path.isdir(BASEPATH + 'train_photos/' + i):
                        n += 1
                # print("n=%s" % (str(n)))
                train(n)
        except Exception as e:
            print(e)
            with open(BASEPATH + 'start.txt', 'w')as f:
                f.write('')
        finally:
            time.sleep(60)



if __name__ == '__main__':
    checkMission()
    # if not os.path.exists(BASEPATH + 'train_photos/'):
    #     os.makedirs(BASEPATH + 'train_photos/')
    #     os.makedirs(BASEPATH + 'model/')
    #     os.makedirs(BASEPATH + 'test_photos/test')
    # for id,t in results:
    #     print(type(t))
    #     if int(t) < int(time.time()) - 600:
    #         pass
    #         # kill jincheng


