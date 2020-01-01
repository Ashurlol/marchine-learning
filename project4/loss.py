import numpy as np
def sum_squares(hyp,y):
    return np.sum((hyp-y)**2)


def cross_entropy(x,y):
    print((x,y), np.sum(y*np.log(x)))
    ''' x should be predictions, y reals'''
    
    return np.sum(y*np.log(x))

        
def zeroone(x, y):
    xClass = 0
    if x[0] < x[1]:
        xClass = 1

    yClass = 0
    if y[0] < y[1]:
        yClass = 1

    if (xClass == yClass):
        return 0
    else:
        return 1
    
