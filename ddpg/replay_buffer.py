from collections import deque
import random
import pickle
import os


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        return

    def load_from_pickle(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.buffer = pickle.load(open(dir_path + '/replay_buffer.pkl', 'rb'))
        return

    def save_to_pickle(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pickle.dump(self.buffer, open(dir_path + '/replay_buffer.pkl', 'wb'))
        return

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def capacity(self):
        return self.capacity

    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        return

    def get_recent_state(self):
        return self.buffer[-1]


if __name__ == '__main__':
    rp = ReplayBuffer(10000)
    rp.load_from_pickle()
    print rp.buffer[0]
    # for i in range(200):
    #     rp.add('state' + str(i), 'action' + str(i), 'reward' + str(i), 'next_state' + str(i), 'done' + str(i))
    # rp.save_to_pickle()
    print rp.size()
