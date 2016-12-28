import tensorflow as tf
import numpy as np
import math
import os
from netutil import *

LEARNING_RATE = 1e-4
TAU = 0.001
L2 = 0.01


class CriticNetwork:

    def __init__(self, sess, state_dim, state_channel, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim
        return

    def create_q_network(self, state_input, action_input):
        # state_dim = self.state_dim
        state_channel = self.state_channel
        action_dim = self.action_dim
        self.state_input = state_input
        self.action_input = action_input

        # conv1
        W_conv1 = weight_variable([8, 8, state_channel, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(state_input, W_conv1, 4) + b_conv1)

        # conv2 = state + action: concat action with conv at 2rd layer
        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # conv3
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'critic: h_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])
        h_fc_action = tf.concat(1, [h_conv3_flat, action_input])

        # fc1
        W_fc1 = weight_variable([h_conv3_out_size + action_dim, 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_fc_action, W_fc1) + b_fc1)

        # output q value, fuck. output is only 1-D, instead of action_dim
        W_fc2 = weight_variable([512, 1])
        b_fc2 = bias_variable([1])
        self.q_value_output = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2)

        self.theta_q = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        return

    def create_target_q_network(self, target_state_input, target_action_input):
        theta_q = self.theta_q

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        update_qt = ema.apply(theta_q)
        theta_qt = [ema.average(x) for x in theta_q]

        h_conv1 = tf.nn.relu(conv2d(target_state_input, theta_qt[0], 4) + theta_qt[1])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, theta_qt[2], 2) + theta_qt[3])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, theta_qt[4], 1) + theta_qt[5])

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'critic: h_target_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])
        h_fc_action = tf.concat(1, [h_conv3_flat, target_action_input])

        h_fc1 = tf.nn.relu(tf.matmul(h_fc_action, theta_qt[6]) + theta_qt[7])
        target_q_value_output = tf.identity(tf.matmul(h_fc1, theta_qt[8]) + theta_qt[9])

        self.target_state_input = target_state_input
        self.target_action_input = target_action_input
        self.target_q_value_output = target_q_value_output
        self.update_qt = update_qt
        self.theta_qt = theta_qt
        return

    def create_training_method(self):
        self.y_input = tf.placeholder('float', [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.theta_q])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        adam = tf.train.AdamOptimizer(LEARNING_RATE)
        grad_var_theta_q = adam.compute_gradients(self.cost, var_list=self.theta_q)
        self.optimizer = adam.apply_gradients(grad_var_theta_q)
        return

    def train(self, y_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })
        return

    def update_target(self):
        self.sess.run(self.update_qt)
        return

    def target_q_value(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    # def q_value(self, state_batch, action_batch):
    #     return self.sess.run(self.q_value_output, feed_dict={
    #         self.state_input: state_batch,
    #         self.action_input: action_batch
    #     })


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    nn = CriticNetwork(sess, 84, 4, 2)
