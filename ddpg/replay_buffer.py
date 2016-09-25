from collections import deque
import random
import pickle


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        return

    def load_from_pickle(self):
        self.buffer = pickle.load(open('replay_buffer.pkl', 'rb'))
        return

    def save_to_pickle(self):
        pickle.dump(self.buffer, open('replay_buffer.pkl', 'wb'))
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


if __name__ == '__main__':
    rp = ReplayBuffer(100)
    rp.load_from_pickle()
    # for i in range(200):
    #     rp.add('state' + str(i), 'action' + str(i), 'reward' + str(i), 'next_state' + str(i), 'done' + str(i))
    # rp.save_to_pickle()
    print rp.size()
