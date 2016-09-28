import numpy as np


def normalize(current, min, max):
    # t = float(current - min) / (max - min)
    mid = (max + min) / 2.0
    span = (max - min) / 2.0
    t = float(current - mid) / span
    return t


def denormalize(current, min, max):
    mid = (max + min) / 2.0
    span = (max - min) / 2.0
    t = current * span + mid
    return t


if __name__ == '__main__':
    print 'env_util'
    # normalize(89, 0, 120)
    t = normalize(10, 0, 120)
    print t
    print denormalize(t, 0, 120)
