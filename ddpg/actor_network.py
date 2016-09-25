import tensorflow as tf
import numpy as np
import math
from netutil import *

INPUT_SIZE = 84
INPUT_CHANNEL = 3
LEARNING_RATE = 1e-6
TAU = 0.001


class ActorNetwork:

    def __init__(self, sess, state_dim, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_input, self.action_output, self.net = self.create_network()
        self.target_state_input, self.target_action_output, self.target_update,\
            self.target_net = self.create_target_network(self.net)

        self.create_training_method()
        self.sess.run(tf.initialize_all_variables())
        self.update_target()
        return

    def create_network(self):
        action_dim = self.action_dim
        # input layer
        state_input = tf.placeholder('float', [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])
        # conv1
        W_conv1 = weight_variable([8, 8, INPUT_CHANNEL, 32])
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
        action_output = tf.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)

        params = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]

        return state_input, action_output, params

    def create_target_network(self, net):
        state_input = tf.placeholder('float', [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        h_conv1 = tf.nn.relu(conv2d(state_input, target_net[0], 4) + target_net[1])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, target_net[2], 2) + target_net[3])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, target_net[4], 1) + target_net[5])

        h_conv3_out_size = np.prod(h_conv3.get_shape().as_list()[1:])
        print 'actor: h_target_conv3_out_size', h_conv3_out_size
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv3_out_size])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, target_net[6]) + target_net[7])
        action_output = tf.tanh(tf.matmul(h_fc1, target_net[8]) + target_net[9])

        return state_input, action_output, target_update, target_net

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder('float', [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.net))
        return

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_batch: state_batch
        })
        return

    def update_target(self):
        self.sess.run(self.target_update)
        return

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })

    def target_actions(self, state_batch):
        return self.sess.run(self.sess, feed_dict={
            self.target_state_input: state_batch
        })

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("models_actor")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print 'Successfully loaded:', checkpoint.model_checkpoint_path
        else:
            print 'Could not find old network weights'
        return

    def save_network(self, time_step):
        print 'save actor-network...', time_step
        self.saver.save(self.sess, 'models_actor/' + 'actor-network', global_step=time_step)
        return


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    nn = ActorNetwork(sess, 84, 4)
