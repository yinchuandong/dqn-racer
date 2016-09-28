from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np
from ddpg.ddpg import DDPG
import ddpg.env_util as EnvUtil


# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='', static_folder='static-ddpg')
app.config['SECRET_KEY'] = 'secret!'
app.debug = False  # you need to cancel debug mode when you run it on gpu
# app.debug = True  # you need to cancel debug mode when you run it on gpu
socketio = SocketIO(app)

# ----------for ddpg-------------------------

STATE_DIM = 84
STATE_CHANNEL = 4
ACTION_DIM = 2

speed_space = [0.0, 12000.0]
playerX_space = [-1.0, 1.0]

ddpgNet = DDPG(STATE_DIM, STATE_CHANNEL, ACTION_DIM)


def getTime():
    return int(round(time.time() * 1000))


@socketio.on('action_space')
def handle_action_space(data):
    print '----------------------handle_action_space------------------------------'
    global speed_space
    global playerX_space
    speed_space = [float(data['speed_space'][0]), float(data['speed_space'][1])]
    playerX_space = [float(data['playerX_space'][0]), float(data['playerX_space'][1])]
    return


@socketio.on('message')
def handle_message(data):
    decode_action = do_train(data)
    emit('action', decode_action)
    return


@app.route('/')
def index_final():
    return app.send_static_file('v4.final.html')


@app.route('/train', methods=['post'])
def req_train():
    data = request.form
    decode_action = do_train(data)
    return jsonify(decode_action)


def do_train(data):
    # print '----------------------------------------------------'
    # image = Image.open(BytesIO(base64.b64decode(data['img']))).convert('RGB')
    image = Image.open(BytesIO(base64.b64decode(data['img']))).convert('L')
    # imgname = 'img/%s.png' % getTime()
    # image.save(imgname)
    start_frame = bool(data['start_frame'])
    print start_frame
    if start_frame or ddpgNet.replay_buffer.size() == 0:
        state = np.stack((image, image, image, image), axis=2)
    else:
        state = ddpgNet.replay_buffer.get_recent_state()[0]

    state = np.stack((image, image, image, image), axis=2)
    playerX = EnvUtil.normalize(float(data['playerX']), playerX_space[0], playerX_space[1])
    speed = EnvUtil.normalize(float(data['speed']), speed_space[0], speed_space[1])
    action = np.asarray([playerX, speed])
    reward = float(data['reward'])
    image = np.reshape(image, (STATE_DIM, STATE_DIM, 1))
    next_state = np.append(image, state[:, :, : (STATE_CHANNEL - 1)], axis=2)
    terminal = bool(data['terminal'])

    # print np.shape(action), action
    # print np.shape(state), np.shape(next_state)
    ddpgNet.perceive(state, action, reward, next_state, terminal)
    action = ddpgNet.action(next_state)
    nextPlayerX = EnvUtil.denormalize(action[0], playerX_space[0], playerX_space[1])
    nextSpeed = EnvUtil.denormalize(action[1], speed_space[0], speed_space[1])
    decode_action = {
        'playerX': nextPlayerX,
        'speed': nextSpeed
    }
    print 'time_step:', ddpgNet.time_step, \
        '/ playerX:', nextPlayerX, \
        '/speed:', nextSpeed
    return decode_action


if __name__ == '__main__':
    socketio.run(app)
