import tensorflow as tf
import numpy as np
import math


class ActorNetwork:

    def __init__(self, sess, state_dim, action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        return

    def create_training_method(self):

        return

    def create_network(self):
        return

    def create_target_network(self, net):

        return

    def update_target(self):
        return

    def train(self):
        return

    def actions(self, state_batch):
        return

    def target_actions(self, state_batch):
        return
