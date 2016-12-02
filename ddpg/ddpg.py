import tensorflow as tf
import numpy as np
import os
import time
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.99


class DDPG:

    def __init__(self, state_dim, state_channel, action_dim):
        self.state_dim = state_dim
        self.state_channel = state_channel
        self.action_dim = action_dim

        self.sess = tf.InteractiveSession()
        self.state_input = tf.placeholder('float', [None, state_dim, state_dim, state_channel])
        self.target_state_input = tf.placeholder('float', [None, state_dim, state_dim, state_channel])
        self.action_input = tf.placeholder('float', [None, action_dim])

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.state_channel, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.state_channel, self.action_dim)

        # create network
        self.actor_network.create_network(self.state_input)
        self.critic_network.create_q_network(self.state_input, self.actor_network.action_output)

        # create target network
        self.actor_network.create_target_network(self.target_state_input)
        self.critic_network.create_target_q_network(self.target_state_input, self.actor_network.target_action_output)

        # create training method
        self.actor_network.create_training_method(self.critic_network.q_value_output)
        self.critic_network.create_training_method()

        self.sess.run(tf.initialize_all_variables())
        self.actor_network.update_target()
        self.critic_network.update_target()

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.exploration_noise = OUNoise(self.action_dim)

        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + '/models_ddpg'
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        # for log
        self.reward_input = tf.placeholder(tf.float32)
        tf.scalar_summary('reward', self.reward_input)
        self.time_input = tf.placeholder(tf.float32)
        tf.scalar_summary('living_time', self.time_input)
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.dir_path + '/log', self.sess.graph)

        self.episode_reward = 0.0
        self.episode_start_time = 0.0

        self.time_step = 1
        self.saver = tf.train.Saver(tf.all_variables())
        self.load_time_step()
        self.load_network()
        return

    def train(self):
        action_dim = self.action_dim

        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)  # sample BATCH_SIZE from replay_buffer
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

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

        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # print np.shape(reward_batch), np.shape(y_batch)

        # train actor network
        self.actor_network.train(state_batch)

        # train critic network
        self.critic_network.train(y_batch, state_batch, action_batch)

        # update target network
        self.actor_network.update_target()
        self.critic_network.update_target()
        return

    def noise_action(self, state):
        action = self.actor_network.action(state)
        return action + self.exploration_noise.noise()

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def _record_log(self, reward, living_time):
        summary_str = self.sess.run(self.summary_op, feed_dict={
            self.reward_input: reward,
            self.time_input: living_time
        })
        self.summary_writer.add_summary(summary_str, self.time_step)
        return

    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.episode_start_time == 0.0:
            self.episode_start_time = time.time()
        # for testing
        # self.time_step += 1
        # if self.time_step == 100:
        #     print '--------------------------------'
        #     self.replay_buffer.save_to_pickle()
        # return
        
        self.episode_reward += reward
        living_time = time.time() - self.episode_start_time
        if self.time_step % 1000 == 0 or done:
            self._record_log(self.episode_reward, living_time)

        if self.replay_buffer.size() > REPLAY_START_SIZE:
            for i in range(10):
                self.train()

        if self.time_step % 100000 == 0:
            self.save_network()

        if done:
            print '===============reset noise========================='
            self.exploration_noise.reset()
            self.episode_reward = 0.0
            self.episode_start_time = time.time()

        self.time_step += 1
        return

    def load_time_step(self):
        if not os.path.exists(self.dir_path):
            return
        files = os.listdir(self.dir_path)
        step_list = []
        for filename in files:
            if ('meta' in filename) or ('-' not in filename):
                continue
            step_list.append(int(filename.split('-')[-1]))
        step_list = sorted(step_list)
        if len(step_list) == 0:
            return
        self.time_step = step_list[-1] + 1
        return

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.dir_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print 'Successfully loaded:', checkpoint.model_checkpoint_path
        else:
            print 'Could not find old network weights'
        return

    def save_network(self):
        print 'save actor-critic network...', self.time_step
        self.saver.save(self.sess, self.dir_path + '/ddpg', global_step=self.time_step)
        return


if __name__ == '__main__':
    ddpg = DDPG(84, 4, 2)
    ddpg.replay_buffer.load_from_pickle()
    # trans = ddpg.replay_buffer.get_recent_state()
    ddpg.train()
    # ddpg.save_network()
    # ddpg.critic_network.save_network(ddpg.time_step)
    # action = ddpg.noise_action(trans[0])
    # print ddpg.time_step
    # print action
    # print trans[1]
    # import env_util as EnvUtil
    # print EnvUtil.denormalize(action[0], -1.0, 1.0), EnvUtil.denormalize(action[1], 0, 12000)
    # print EnvUtil.denormalize(trans[1][0], -1.0, 1.0), EnvUtil.denormalize(trans[1][1], 0, 12000)
