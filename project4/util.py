import math
import numpy as np
def rand_mat(r,c):
    return 2*(np.random.rand(r,c) - 0.5)

def diff(lossfn, y, x, k, delta=1e-12):
    cached = lossfn(y,x)
    x[k] += delta
    update = lossfn(y,x)
    x[k] -= delta
    return (update-cached)/delta


def load_data(filename):
    data = []
    with open(filename) as fh:
        for line in fh:
            if line[0] != '#':
                x = eval(line)[0][0]
                y = eval(line)[1][0]
                data.append((x,y))
                
    return data


def load_data(filename):
    data = []
    with open(filename) as fh:
        for line in fh:
            (x,y) = line.strip().split()
            data.append((float(x), float(y)))
    return data
    

def ce(p, q):
    assert(len(p) == len(q))

    acc = 0.0
    for n in range(len(p)):
        for k in range(len(p[n])):
            for j in range(len(p[n][k])):
                acc += -p[n][k][j]*math.log(q[n][k][j])
    return acc
            
