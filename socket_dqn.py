from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np
from dqn import DQN
from agent import Agent
import os

# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='', static_folder='static-dqn')
app.config['SECRET_KEY'] = 'secret!'
# app.debug = False  # you need to cancel debug mode when you run it on gpu
app.debug = True  # you need to cancel debug mode when you run it on gpu
socketio = SocketIO(app)
dqnnet = DQN()
agent = Agent()


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
    print '----------------------------------------------------'
    # print msg
    # print msg['status']
    # print msg['telemetry']
    image = Image.open(BytesIO(base64.b64decode(msg['img']))).convert('RGB')
    # imgname = 'img/%s.png' % getTime()
    # image.save(imgname)
    image_arr = np.asarray(image)
    # print np.shape(image_arr)
    left, right, faster, slower = msg['action']
    action = agent.encode_action(left, right, faster, slower)
    terminal = msg['terminal']
    start_frame = msg['start_frame']
    telemetry = msg['telemetry']
    reward = agent.get_mean_reward(telemetry)

    # observe the transition
    dqnnet.perceive(image_arr, action, reward, terminal, start_frame, telemetry)
    dqnnet.train_Q_network()
    recent_state = dqnnet.transition.get_recent_state()
    # 1. for training
    # action_id, max_q_value = dqnnet.epsilon_greedy(recent_state)
    # 2. for testing
    action_id, max_q_value = dqnnet.get_action_index(recent_state)
    dqnnet.print_info(action_id, reward, max_q_value)
    # print action_id, agent.decode_action(action_id)
    new_action = agent.decode_action(action_id)
    emit('message', new_action)
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
