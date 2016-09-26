from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np

# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='', static_folder='static-ddpg')
app.config['SECRET_KEY'] = 'secret!'
# app.debug = False  # you need to cancel debug mode when you run it on gpu
app.debug = True  # you need to cancel debug mode when you run it on gpu
socketio = SocketIO(app)



@socketio.on('message')
def handle_message(msg):
    print '----------------------------------------------------'
    # print msg
    # print msg['status']
    # print msg['telemetry']
    image = Image.open(BytesIO(base64.b64decode(msg['img']))).convert('RGB')
    # imgname = 'img/%s.png' % getTime()
    # image.save(imgname)
    image_arr = np.asarray(image)
    print np.shape(image_arr)
    return


@app.route('/final')
def index_final():
    return app.send_static_file('v4.final.html')


@app.route('/hills')
def index_hill():
    return app.send_static_file('v3.hills.html')


@app.route('/test')
def test():
    return 'test1'


if __name__ == '__main__':
    socketio.run(app)
