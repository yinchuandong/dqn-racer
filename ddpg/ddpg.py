import tensorflow as tf
import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise

BATCH_SIZE = 32

class DDPG:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.sess = tf.InteractiveSession()
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

        self.exploration_nose = OUNoise(self.action_dim)
        return

    def train(self):
        minibatch = []  # sample BATCH_SIZE from replay_buffer 
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # if action_dim = 1, it's a number not a array
        action_batch = np.resize([BATCH_SIZE, action_dim])
        return

    def noise_action(self, state):

        return

    def action(self, state):
        return

    def perceive(self, state, action, reward, next_state, done):

        return

if __name__ == '__main__':
    ddpg = DDPG(84, 2)
