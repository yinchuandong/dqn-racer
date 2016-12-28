
def normalize(current, minNum, maxNum):
    '''
    return a float number from -1.0 to 1.0
    '''
    mid = (maxNum + minNum) / 2.0
    span = (maxNum - minNum) / 2.0
    t = float(current - mid) / span
    t = max(-1.0, min(t, 1.0))
    return t


def denormalize(current, minNum, maxNum):
    mid = (maxNum + minNum) / 2.0
    span = (maxNum - minNum) / 2.0
    t = current * span + mid
    t = min(maxNum, max(minNum, t))
    return t


if __name__ == '__main__':
    print 'env_util'
    # normalize(89, 0, 120)
    t = normalize(-509, -500, 500)
    print t
    # print denormalize(t, -500, 500)
    print denormalize(t, -500, 500)
