import numpy as np


def normalize(current, min, max):
    t = float(current - min) / (max - min)
    return t


def denormalize(current, min, max):
    t = current * (max - min) + min
    return t


if __name__ == '__main__':
    print 'env_util'
    # normalize(89, 0, 120)
    t = normalize(120, -120, 120)
    # print t
    print denormalize(t, -120, 120)
