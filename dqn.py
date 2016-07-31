# encoding=utf-8

import tensorflow as tf
import numpy as np
import sys
import random
from config import *
from transition import Transistion


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


def output_size(in_size, filter_size, stride):
    return (in_size - filter_size) / stride + 1


class DQN(object):

    def __init__(self):
        self.timesteps = 0
        self.transition = Transistion()  # replay buffer: D
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
        self.saver = tf.train.Saver(tf.all_variables())
        checkpoint = tf.train.get_checkpoint_state('models')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print 'Successfully loaded:', checkpoint.model_checkpoint_path
        else:
            print 'Could not find old network weights'

        return

    def create_network(self):
        # input layer
        s = tf.placeholder('float', shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], name='s')

        # hidden conv layer
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([4, 4, 32, 64], name='W_conv2')
        b_conv2 = bias_variable([64], name='b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

        W_conv3 = weight_variable([3, 3, 64, 64], name='W_conv3')
        b_conv3 = bias_variable([64], name='b_conv3')
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size], name='h_conv3_flat')

        W_fc1 = weight_variable([h_conv3_out_size, 512], name='W_fc1')
        b_fc1 = bias_variable([512], name='b_fc1')
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer: Q_value
        W_fc2 = weight_variable([512, ACTIONS], name='W_fc2')
        b_fc2 = bias_variable([ACTIONS], name='b_fc2')
        Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.s = s
        self.Q_value = Q_value
        return

    def create_update(self):
        self.a = tf.placeholder('float', shape=[None, ACTIONS], name='a')
        self.y = tf.placeholder('float', shape=[None], name='y')
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - Q_action))
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        return

    def perceive(self, image, action, reward, terminal, start_frame, telemetry):
        self.transition.add(image, action, reward, terminal, start_frame, telemetry)
        return

    def get_action_index(self, state):
        """ use it in test phase
        :param state: 1x84x84x3
        """
        Q_value_t = self.Q_value.eval(feed_dict={self.s: state})[0]
        return np.argmax(Q_value_t)

    def epsilon_greedy(self, state):
        """
        :param state: 1x84x84x3
        """
        Q_value_t = self.Q_value.eval(feed_dict={self.s: state})[0]
        action_index = 0
        if random.random() <= self.epsilon:
            print '------------random action---------------'
            action_index = random.randrange(ACTIONS)
        else:
            action_index = np.argmax(Q_value_t)

        if self.epsilon > FINAL_EPSILON and self.timesteps > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        max_q_value = np.max(Q_value_t)
        return action_index, max_q_value

    def train_Q_network(self):
        self.timesteps += 1
        if (self.timesteps <= OBSERVE):
            return
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.transition.get_minibatch()

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.s: next_state_batch})
        for i in range(0, BATCH_SIZE):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y: y_batch,
            self.a: action_batch,
            self.s: state_batch
        })

        self.save_model()
        return

    def print_info(self, action_id, reward, q_value):
        state = ''
        if self.timesteps <= OBSERVE:
            state = 'observer'
        elif self.timesteps > OBSERVE and self.timesteps <= OBSERVE + EXPLORE:
            state = 'explore'
        else:
            state = 'train'
        print 'timesteps:', self.timesteps, '/ state:', state, '/ epsilon:', self.epsilon, \
            '/ action:', action_id, '/ reward:', reward, '/ q_value:', q_value
        return

    def save_model(self):
        if self.timesteps % 10000 == 0:
            self.saver.save(self.session, './models/' + GAME + '-dqn', global_step=self.timesteps)
        return


def test():
    print 'test----'
    # dqnModel = DQN()
    conv1 = output_size(84, 8, 4)
    conv2 = output_size(conv1, 4, 2)
    conv3 = output_size(conv2, 3, 1)
    print conv1, conv2, conv3
    print (conv3 ** 2) * 64
    return

if __name__ == '__main__':
    test()
