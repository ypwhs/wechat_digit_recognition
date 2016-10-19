#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import cgitb
import time
import requests
import urlparse
from wechatpy import parse_message
from wechatpy.utils import check_signature
from wechatpy.exceptions import InvalidSignatureException
from wechatpy.replies import create_reply, ArticlesReply

import time
import cv2
import numpy as np
from keras.models import model_from_json

token='your token'
encoding_aes_key='your key'


def resize(rawimg):  # resize img to 28*28
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = (28 - w) / 2
    y = (28 - h) / 2
    outimg[y:y+h, x:x+w] = img
    return outimg


def convert(imgpath):   # read digits
    img = cv2.imread(imgpath)
    gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 25)
    img2, ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    for rect in rects:
        x, y, w, h = rect
        roi = gray[y:y+h, x:x+w]
        hw = float(h) / w
        if (w < 200) & (h < 200) & (h > 10) & (w > 10) & (1.1 < hw) & (hw < 5):
            res = resize(roi)
            res = cv2.bitwise_not(res)
            res = np.resize(res, (1, 784))

            predictions = model.predict(res)
            predictions = np.argmax(predictions)
            if predictions != 10:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(img, '{:.0f}'.format(predictions), (x, y), cv2.FONT_HERSHEY_DUPLEX, h/25.0, (255, 0, 0))
    return img


# 启用调试
cgitb.enable()

# 获取 POST 内容
body_text = sys.stdin.read()

# 如果含参数，解析各个参数
print "Content-Type: text/html"
print ""

# 获取 URL 参数
query_string = os.environ.get("QUERY_STRING")

if query_string == '':
    print '本页面仅允许微信访问'
    sys.exit(0)

try:
    arguments = urlparse.parse_qs(query_string)
    signature = arguments['signature'][0]
    timestamp = arguments['timestamp'][0]
    nonce = arguments['nonce'][0]
except:
    print 'arguments error'
    sys.exit(0)

# 校验时间戳。5 分钟以前的 timestamp 自动拒绝
current_timestamp = int(time.time())

if (current_timestamp - int(timestamp)) > 300:
    print 'Incorrect timestamp'
    sys.exit(0)

# 接口检测部分
try:
    check_signature(token, signature, timestamp, nonce)
except InvalidSignatureException:
    print 'error'
    sys.exit(0)

if 'echostr' in arguments:
    echostr = arguments['echostr'][0]
    print echostr
    sys.exit(0)

msg = parse_message(body_text)
reply = ''
if msg.type == 'text':
    reply = create_reply('Text:' + msg.content.encode('utf-8'), message=msg)
elif msg.type == 'image':
    reply = create_reply('图片', message=msg)
    try:
        r = requests.get(msg.image) # download image
        filename = 'img/' + str(int(time.time())) + '.jpg';
        convertfilename = filename.replace('.', '.convert.')
        with open(filename, 'w') as f:
            f.write(r.content)
        if cv2.imread(filename) is not None:
            # load model
            with open('model.json', 'r') as f:
                model = model_from_json(f.read())
            model.load_weights('model.h5')
            
            cv2.imwrite(convertfilename, convert(filename))
            url = 'http://w.luckiestcat.com/' + convertfilename
            reply = ArticlesReply(message=msg, articles=[{
                'title': u'识别成功',
                'url': url,
                'description': u'',
                'image': url
            }])
    except:
        reply = create_reply('识别失败', message=msg)

print reply
