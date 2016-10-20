import numpy as np
import matplotlib.pyplot as plt



def draw_q_value():
    matrix = np.loadtxt('statistic/q_value.txt', delimiter=',')
    # matrix = matrix[0:10, :]
    x = matrix[:, 0]
    y = matrix[:, 1]
    plt.xlabel('timestep')
    plt.ylabel('target q value')
    plt.plot(x, y)
    plt.show()
    return


def draw_lap_time():
    matrix = np.loadtxt('statistic/game.txt', delimiter=',')
    matrix = matrix[:1000, :]
    x = matrix[:, 0]
    y = matrix[:, 2]
    plt.xlabel('timestep')
    plt.ylabel('lap time')
    plt.plot(x, y)
    plt.show()
    return


def draw_action_q_value():
    matrix = np.loadtxt('statistic/game.txt', delimiter=',')
    matrix = matrix[:1000, :]
    x = matrix[:, 0]
    y = matrix[:, 3]
    plt.xlabel('timestep')
    plt.ylabel('target q value')
    plt.plot(x, y)
    plt.show()
    return

if __name__ == '__main__':
    draw_q_value()
    draw_lap_time()
    draw_action_q_value()



