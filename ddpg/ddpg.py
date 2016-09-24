import tensorflow as tf
import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise

BATCH_SIZE = 32
GAMMA = 0.99

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

        # calculate y_batch via target network
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])

        # train critic network
        self.critic_network.train(y_batch, state_batch, action_batch)

        # update the actor policy using the sampled gradients
        # get the graident of action and pass it to network
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        # train actor network
        self.actor_network.train(q_gradient_batch, state_batch)

        # update target network
        self.actor_network.update_target()
        self.critic_network.update_target()
        return

    def noise_action(self, state):

        return

    def action(self, state):
        return

    def perceive(self, state, action, reward, next_state, done):

        return

if __name__ == '__main__':
    ddpg = DDPG(84, 2)
    ddpg.train()
