# encoding=utf-8

import random


class Agent(object):

    def __init__(self):
        # (left, right, faster, slower)
        action_left = (True, False, True, False)
        action_right = (False, True, True, False)
        action_faster = (False, False, True, False)
        self.action_id_dict = dict()
        self.action_id_dict[action_left] = 0
        self.action_id_dict[action_right] = 1
        self.action_id_dict[action_faster] = 2
        self.id_action_dict = dict()
        self.id_action_dict[0] = action_left
        self.id_action_dict[1] = action_right
        self.id_action_dict[2] = action_faster
        return

    def encode_action(self, left, right, faster, slower):
        if left and right:
            raise ValueError("Invalid action, cannot press both left and right")
        if faster and slower:
            raise ValueError("Invalid action, cannot press both faster and slower")
        return self.action_id_dict[(left, right, faster, slower)]

    def decode_action(self, action_id):
        left, right, faster, slower = self.id_action_dict[action_id]
        return {
            'keyLeft': left,
            'keyRight': right,
            'keyFaster': faster,
            'keySlower': slower
        }

    def random_action(self):
        return random.randint(0, 2)

    def reward_frame(self, frame):
        posX = abs(frame['player_x'])
        if frame['collision']:
            return -1.0
        elif posX > 0.8:
            return -0.8
        elif float(frame['speed']) == 0:
            return -1.0
        else:
            is_in_lane = posX <= 0.2 or (posX >= 0.5 and posX <= 0.8)
            penalty = 1.0 if is_in_lane else 0.8
            return penalty * (float(frame['speed']) / float(frame['max_speed']))

    def get_mean_reward(self, telemetry):
        """ calculate the avg reward of telemetry array
        """
        return sum([self.reward_frame(frame) for frame in telemetry]) / float(len(telemetry))

if __name__ == '__main__':
    a = Agent()
    id = a.encode_action(False, True, True, False)
    json = a.decode_action(id)
    print id
    print json
    print a.random_action()
