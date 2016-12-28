import tensorflow as tf
import numpy as np
import math
import os
from netutil import *

LEARNING_RATE = 1e-4
TAU = 0.001


class ActorNetwork:

    def __init__(self, sess, state_dim, state_channel, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim
        return

    def create_network(self, state_input):
        # state_dim = self.state_dim
        state_channel = self.state_channel
        action_dim = self.action_dim
        self.state_input = state_input

        # conv1
        W_conv1 = weight_variable([8, 8, state_channel, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(state_input, W_conv1, 4) + b_conv1)

        # conv2
        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # conv3
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'actor: h_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        # fc1
        W_fc1 = weight_variable([h_conv3_out_size, 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # fc2
        W_fc2 = weight_variable([512, action_dim])
        b_fc2 = bias_variable([action_dim])
        self.action_output = tf.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)

        self.theta_p = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        return

    def create_target_network(self, target_state_input):
        theta_p = self.theta_p
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        update_pt = ema.apply(theta_p)
        theta_pt = [ema.average(x) for x in theta_p]

        h_conv1 = tf.nn.relu(conv2d(target_state_input, theta_pt[0], 4) + theta_pt[1])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, theta_pt[2], 2) + theta_pt[3])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, theta_pt[4], 1) + theta_pt[5])

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'actor: h_target_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, theta_pt[6]) + theta_pt[7])
        target_action_output = tf.tanh(tf.matmul(h_fc1, theta_pt[8]) + theta_pt[9])

        self.target_state_input = target_state_input
        self.target_action_output = target_action_output
        self.theta_pt = theta_pt
        self.update_pt = update_pt
        return

    def create_training_method(self, q_value_input):
        self.q_value_input = q_value_input
        # can add weight decay later
        self.cost = -tf.reduce_mean(self.q_value_input)
        adam = tf.train.AdamOptimizer(LEARNING_RATE)
        # separate compute and apply, because q_value has theta_p and theta_q
        # here we just calculate partial theta_p
        grad_var_theta_p = adam.compute_gradients(self.cost, var_list=self.theta_p)
        self.optimizer = adam.apply_gradients(grad_var_theta_p)
        return

    def train(self, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.state_input: state_batch
        })
        return

    def update_target(self):
        self.sess.run(self.update_pt)
        return

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    nn = ActorNetwork(sess, 84, 4, 2)
    nn.save_network(0)
