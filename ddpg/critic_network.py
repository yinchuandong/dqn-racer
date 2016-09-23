import tensorflow as tf
import numpy as np
import math


class CriticNetwork:

    def __init__(self, sess, state_dim, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        return

    def create_training_method(self):

        return

    def create_q_network(self):
        return

    def create_target_q_network(self, net):

        return

    def update_target(self):
        return

    def train(self, y_batch, state_batch, action_batch):
        return

    def gradients(self, state_batch, action_batch):
        return

    def target_q(self, state_batch):
        return

    def q_value(self, state_batch, action_batch):
        return
