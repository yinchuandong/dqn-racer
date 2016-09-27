import tensorflow as tf
import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 100
BATCH_SIZE = 3
GAMMA = 0.99


class DDPG:

    def __init__(self, state_dim, state_channel, action_dim):
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim

        self.sess = tf.InteractiveSession()
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.state_channel, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.state_channel, self.action_dim)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.exploration_nose = OUNoise(self.action_dim)

        self.time_step = 1
        return

    def train(self):
        action_dim = self.action_dim
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)  # sample BATCH_SIZE from replay_buffer
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # print np.shape(next_state_batch)
        # print next_state_batch[0]
        # return

        # if action_dim = 1, it's a number not a array
        action_batch = np.resize(action_batch, [BATCH_SIZE, action_dim])

        # calculate y_batch via target network
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q_value(next_state_batch, next_action_batch)

        y_batch = []
        for i in range(BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, action_dim])

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
        action = self.actor_network.action(state)
        return action + self.exploration_nose.noise()

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.size() > REPLAY_START_SIZE:
            self.train()
            # self.replay_buffer.save_to_pickle()

        if self.time_step % 10000 == 0:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)

        if done:
            self.exploration_nose.reset()
        return


if __name__ == '__main__':
    ddpg = DDPG(84, 4, 2)
    ddpg.replay_buffer.load_from_pickle()
    ddpg.train()
