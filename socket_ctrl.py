from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import time

# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
app.debug = True
socketio = SocketIO(app)


def getTime():
    return int(round(time.time() * 1000))


@socketio.on('init')
def handle_init(msg):
    print msg
    action = {'keyLeft': False, 'keyRight': False, 'keyFaster': True, 'keySlower': False}
    emit('init', action)
    return


@socketio.on('message')
def handle_message(msg):
    # send('received message: ' + message)
    print 'received message: '
    # print msg
    print msg['status']
    image = Image.open(BytesIO(base64.b64decode(msg['img'])))
    imgname = 'img/%s.png' % getTime()
    image.save(imgname)
    # action = agent.step(transition)
    # return action
    return


@app.route('/')
def index():
    return app.send_static_file('v4.final.html')


@app.route('/test')
def test():
    return 'test1'

if __name__ == '__main__':
    socketio.run(app)


















