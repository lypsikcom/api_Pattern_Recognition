# -*- coding:utf8 -*-
import sys
import io
from app import app
from checkmission import checkMission
from multiprocessing import Process
import os

import warnings


warnings.filterwarnings("ignore")
sys.path.append(".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main(count=0):
    count += 1
    if count > 3:
        raise Exception("Program startup failure! 3 times! Error!")
    try:
        p = Process(target=checkMission)
        p.daemon = True  # 一定要在p.start()前设置,设置p为守护进程,禁止p创建子进程,并且父进程代码执行结束,p即终止运行
        p.start()
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
        # os.system("gunicorn -c gunicorn.py -p slavett.pid --graceful-timeout 30 app:app")  # linux生产
    except Exception as e:
        main(count)

if __name__ == '__main__':
    main()

