import tensorflow as tf
import numpy as np
import math
from netutil import *

INPUT_SIZE = 84
INPUT_CHANNEL = 3
LEARNING_RATE = 1e-6
TAU = 0.001
BATCH_SIZE = 64


class CriticNetwork:

    def __init__(self, sess, state_dim, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_input, self.action_input, self.q_value_output, self.net = self.create_q_network()
        return

    def create_q_network(self):
        action_dim = self.action_dim
        # state_dim = self.state_dim

        # input layer
        state_input = tf.placeholder('float', [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])
        action_input = tf.placeholder('float', [None, action_dim])

        # conv1
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(state_input, W_conv1, 4) + b_conv1)
        h_conv1_out_size = np.prod(h_conv1.get_shape().as_list()[1:])

        # conv2 = state + action: concat action with conv at 2rd layer
        W_conv2 = weight_variable([4, 4, 32, 64])
        W_action2 = self.variable([self.action_dim, 64], h_conv1_out_size + action_dim)
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + tf.matmul(action_input, W_action2) + b_conv2)

        # conv3
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'h_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        # fc1
        W_fc1 = weight_variable([h_conv3_out_size, 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # output
        W_fc2 = weight_variable([512, action_dim])
        b_fc2 = bias_variable([action_dim])
        q_value_output = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2)

        params = [W_conv1, b_conv1, W_conv2, W_action2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        return state_input, action_input, q_value_output, params

    def create_target_q_network(self, net):

        return

    def create_training_method(self):
        return

    def train(self, y_batch, state_batch, action_batch):
        return

    def update_target(self):
        return

    def gradients(self, state_batch, action_batch):
        return

    def target_q(self, state_batch):
        return

    def q_value(self, state_batch, action_batch):
        return

    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    nn = CriticNetwork(sess, 84, 3)
