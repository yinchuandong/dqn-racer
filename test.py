from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def testImageChannel():
    image = Image.open('img/1469931653058.png').convert('RGB')
    arr = np.asarray(image)
    print np.shape(arr)
    plt.imshow(arr)
    plt.show()
    return


if __name__ == '__main__':
    testImageChannel()