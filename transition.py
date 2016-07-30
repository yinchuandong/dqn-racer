import numpy as np
from collections import deque
from dqn import REPLAY_MEMORY, BATCH_SIZE, INPUT_SIZE, INPUT_CHANNEL


class Transistion(object):

    def __init__(self):
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY)
        return

    def add(self, image, action, reward, terminal, start_frame, telemetry):
        """ this method is made to add a transition for experience replay
        :param image: {numpy array}, width * height * channel (80x80x3)
        :param action: {integer}, encoded from (left, right, faster, slower)
        :param reward: {float} from -10.0 to 10, cacluated by {telemetry}
        :param terminal: {bool} whether game is terminated
        :param start_frame: {bool} wheter this frame is the first frame
        :param telemetry: {dict} status of the car, including (positionX, speed, maxSpeed)
        """
        t = (image, action, reward, terminal, start_frame, telemetry)
        self.replay_buffer.append(t)
        return

    def get_frame_by_id(self, index):
        cur_trans = self.replay_buffer[index]
        action = cur_trans[1]
        reward = cur_trans[2]
        terminal = cur_trans[3]

        next_state = cur_trans[0]
        start_frame = cur_trans[4]
        if not start_frame:
            state = self.replay_buffer[index - 1][0]
        else:
            state = cur_trans[0]
        return (state, action, reward, next_state, terminal)

    def get_minibatch(self):
        # sample from experience replay
        state_batch = np.zeros((BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
        action_batch = np.zeros(BATCH_SIZE)
        reward_batch = np.zeros(BATCH_SIZE)
        next_state_batch = np.zeros_like(state_batch)
        terminal_batch = np.zeros(BATCH_SIZE)

        for i in range(BATCH_SIZE):
            action_index = np.random.randint(0, len(self.replay_buffer))
            state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i],
            terminal_batch[i] = self.get_frame_by_id(action_index)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def get_recent_state(self):
        state = self.replay_buffer[-1][0]
        return state.reshape((1, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))

if __name__ == '__main__':
    # tran = Transistion()
    print int(False)
    # print 1
