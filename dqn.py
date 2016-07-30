# encoding=utf-8

import tensorflow as tf
import numpy as np
import PIL
import sys
import random
from collections import deque

GAME = 'racer'
INPUT_SIZE = 84
INPUT_CHANNEL = 3
ACTIONS = 3
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FRAME_PER_ACTION = 1


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class DQN(object):

    def __init__(self):
        self.timesteps = 0
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY)  # replay buffer: D
        self.epsilon = INITIAL_EPSILON

        # q-network parameter
        self.s = None
        self.Q_value = None
        self.a = None  # action input
        self.y = None  # y input
        self.cost = None  # loss function
        self.optimizer = None  # tensorflow AdamOptimizer

        self.create_network()
        self.create_update()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # resotre from checkpoint
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state('models')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print 'Successfully loaded:', checkpoint.model_checkpoint_path
        else:
            print 'Could not find old network weights'
        return

    def create_network(self):
        # input layer
        s = tf.placeholder('float', [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], name='s')

        # hidden conv layer
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)

        W_conv2 = weight_variable([4, 4, 32, 64], name='W_conv2')
        b_conv2 = bias_variable([64], name='b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        W_conv3 = weight_variable([3, 3, 64, 64], name='W_conv3')
        b_conv3 = bias_variable([64], name='b_conv3')
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1), + b_conv3)
        # h_conv3: [batch, w, h, feature], output = w * h *feature
        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size], name='h_conv3_flat')

        W_fc1 = weight_variable([h_conv3_out_size, 512], name='W_fc1')
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer: Q_value
        W_fc2 = weight_variable([512, ACTIONS], name='W_fc2')
        b_fc2 = bias_variable([ACTIONS], name='b_fc2')
        Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.s = s
        self.Q_value = Q_value
        return

    def create_update(self):
        self.a = tf.placeholder('float', [None, ACTIONS])
        self.y = tf.placeholder('float', [ACTIONS])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - Q_action))
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        return

    def perceive(self, state, action_index, reward, next_state, terminal):
        action = np.zeros([ACTIONS])
        action[action_index] = 1
        self.replay_buffer.append((state, action, reward, next_state, terminal))

        if self.timesteps > OBSERVE:
            self.train_Q_network()
        return

    def get_action_index(self, state):
        # use it in test phase
        Q_value_t = self.Q_value.eval(feed_dict={self.s: [state]})[0]
        return np.argmax(Q_value_t)

    def epsilon_greedy(self, state):
        Q_value_t = self.Q_value.eval(feed_dict={self.s: [state]})[0]
        action_index = 0
        if random.random() <= self.epsilon:
            print '------------random action---------------'
            action_index = random.randrange(ACTIONS)
        else:
            action_index = np.argmax(Q_value_t)

        if self.epsilon > FINAL_EPSILON and self.timesteps > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action_index

    def train_Q_network(self):
        self.timesteps += 1
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [d[0] for d in minibatch]
        action_batch = [d[1] for d in minibatch]
        reward_batch = [d[2] for d in minibatch]
        next_state_batch = [d[3] for d in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.s: next_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y: y_batch,
            self.a: action_batch,
            self.s: state_batch
        })
        return


def init():

    return


def main():

    return

if __name__ == '__main__':
    main()
