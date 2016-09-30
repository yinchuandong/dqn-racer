import tensorflow as tf
import numpy as np
import math
import os
from netutil import *

LEARNING_RATE = 1e-6
TAU = 0.001
L2 = 0.01


class CriticNetwork:

    def __init__(self, sess, state_dim, state_channel, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim

        self.state_input, self.action_input, self.q_value_output, self.net = self.create_q_network()
        self.target_state_input, self.target_action_input, self.target_q_value_output,\
            self.target_update = self.create_target_q_network(self.net)

        self.create_training_method()
        self.sess.run(tf.initialize_all_variables())
        self.update_target()
        return

    def create_q_network(self):
        state_dim = self.state_dim
        state_channel = self.state_channel
        action_dim = self.action_dim

        # input layer
        state_input = tf.placeholder('float', [None, state_dim, state_dim, state_channel])
        action_input = tf.placeholder('float', [None, action_dim])

        # conv1
        W_conv1 = weight_variable([8, 8, state_channel, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(state_input, W_conv1, 4) + b_conv1)
        h_conv1_out_size = np.prod(h_conv1.get_shape().as_list()[1:])

        # conv2 = state + action: concat action with conv at 2rd layer
        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        W_action2 = self.variable([self.action_dim, 64], h_conv1_out_size + action_dim)
        h_action2 = tf.reshape(tf.matmul(action_input, W_action2), [-1, 1, 1, 64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + h_action2 + b_conv2)

        # conv3
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'critic: h_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        # fc1
        W_fc1 = weight_variable([h_conv3_out_size, 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # output q value, fuck. output is only 1-D, instead of action_dim
        W_fc2 = weight_variable([512, 1])
        b_fc2 = bias_variable([1])
        q_value_output = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2)

        params = [W_conv1, b_conv1, W_conv2, W_action2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        return state_input, action_input, q_value_output, params

    def create_target_q_network(self, net):
        state_dim = self.state_dim
        state_channel = self.state_channel
        action_dim = self.action_dim

        state_input = tf.placeholder('float', [None, state_dim, state_dim, state_channel])
        action_input = tf.placeholder('float', [None, action_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        h_conv1 = tf.nn.relu(conv2d(state_input, target_net[0], 4) + target_net[1])
        h_action2 = tf.reshape(tf.matmul(action_input, target_net[3]), [-1, 1, 1, 64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, target_net[2], 2) + h_action2 + target_net[4])
        # h_conv2 = tf.nn.relu(conv2d(h_conv1, target_net[2], 2) +
        # tf.matmul(action_input, target_net[3]) + target_net[4])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, target_net[5], 1) + target_net[6])

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'critic: h_target_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, target_net[7]) + target_net[8])
        q_value_output = tf.identity(tf.matmul(h_fc1, target_net[9]) + target_net[10])
        return state_input, action_input, q_value_output, target_update

    def create_training_method(self):
        self.y_input = tf.placeholder('float', [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)
        return

    def train(self, y_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })
        return

    def update_target(self):
        self.sess.run(self.target_update)
        return

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_q_value(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    nn = CriticNetwork(sess, 84, 4, 2)
