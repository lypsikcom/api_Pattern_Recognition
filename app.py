import os
import sys
import time

import requests
from flask import Flask, request, json
from Setting import BASEPATH
from test_data import testPic
from Logger import log
import warnings


warnings.filterwarnings("ignore")
sys.path.append('.')
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/',methods=['GET'])
def index():
    return 'Pattern Recognition!'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            responseDict = {}
            files = request.files.get('file')
            if files:
                savepath = BASEPATH + 'train.zip'
                with open(savepath, 'wb') as f:
                    data = files.read()
                    f.write(data)
                # 添加这个任务到本地
                with open(BASEPATH + 'start.txt', 'w')as f:
                    f.write('')
                responseDict = {"msg": "Success!", "resultCode": "1"}
            else:
                responseDict = {"msg": "No Files!", "resultCode": "5"}
        except Exception as e:
            responseDict = {"msg": "Error!", "resultCode": "0"}
            log.logger.info("upload(error):error!Exception = %s" % (str(e)))
        log.logger.info("upload(end):Success!result = %s" % (str(json.dumps(responseDict))))
        return json.dumps(responseDict, ensure_ascii=False)

@app.route('/recognition', methods=['POST'])
def recognition():
    if request.method == 'POST':
        start = time.time()
        result_dict = {'msg':'Error!', "resultCode":'0'}
        # files = request.files.get('file')
        url = request.form.get('url')
        log.logger.info("recognition(start):start:url = %s" % (str(url)))
        if url:
            res = requests.get(url)
            files = res.content
            if not os.path.exists(BASEPATH + 'ok.txt'):
                result_dict = {'msg': 'Training,Wait!', "resultCode": '2'}
            else:
                if files:
                    # deleteFiles(BASEPATH + 'test_photos/test/')
                    picname = 'pic' + str(int(time.time() * 100)) + '.jpg'
                    savepath = BASEPATH + 'test_photos/test/' + picname
                    with open(savepath, 'wb') as f:
                        f.write(files)
                    try:
                        result = testPic(savepath)
                        result_dict = {'msg': 'Successed!', "resultCode": '1', "result": result}
                    except Exception as e:
                        result_dict = {'msg': 'Error!', "resultCode": '0'}
                        log.logger.info("recognition(error):Recognition Error!Exception = %s" % (str(e)))
                    try:
                        os.remove(savepath)
                    except Exception as e:
                        log.logger.info("recognition(error):Remove Error!Exception = %s" % (str(e)))
                else:
                    result_dict = {'msg': 'File Not Exist', "resultCode": '3'}
        else:
            result_dict = {'msg': 'Parmas Error!', "resultCode": '5'}
        end = time.time()
        log.logger.info("recognition(end):Time=%.3f (S)Success!result = %s" % ((end - start), str(json.dumps(result_dict))))
        return json.dumps(result_dict, ensure_ascii=False)



def deleteFiles(path,flag=0):
    for i in os.listdir(path):
        if path[-1] != '/' or path[-1] != '\\':
            path = path + '/'
        file_data = path + i
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            deleteFiles(file_data, flag=1)
    if flag:
        os.rmdir(path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
