import math
import numpy as np

leak = 0.03
def relu(x):
    if x > 0:
        return x
    else:
        return leak*x

def d_relu(x):
    if x > 0:
        return 1
    else:
        return leak

#def sigmoid(x):
#    return 1 / (1 + math.e**-x)

# more robust to large values
def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s*(1-s)

def softmax(x):
    '''
    :param x: - a numpy array
    '''
    x = np.exp(x-np.max(x))
    # stability tweak fro https://deepnotes.io/softmax-crossentropy
    return x/np.sum(x)
