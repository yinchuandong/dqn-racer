from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import base64
import time
import numpy as np
from dqn import DQN
from agent import Agent

# python xxx.py
# 127.0,0.1:5000
app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
app.debug = True
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
    # size = len(dqnnet.transition.replay_buffer)
    # print dqnnet.transition.replay_buffer[-1][1:5]
    dqnnet.train_Q_network()
    recent_img = dqnnet.transition.get_recent_state()
    print np.shape(recent_img)
    action_id = dqnnet.epsilon_greedy(recent_img)

    dqnnet.print_info()
    print action_id, agent.decode_action(action_id)

    return


@app.route('/')
def index():
    return app.send_static_file('v4.final.html')


@app.route('/test')
def test():
    return 'test1'

if __name__ == '__main__':
    socketio.run(app)
