from time import time
import sys
import pickle
import numpy as np
import random
import math
import activations
from loss import sum_squares
from util import *

bptime=0
bpcalls=0
tgrad = 0

class MLP:
    '''An ADT-style implementation of a Multilayer Perceptron.

    Slots:
        w:  an array of weight matrices
        L:  the number of hidden layers
        k:  the number of units per layer (including hidden, input, and output)
        a:  cached unit activations (for backprop)
        z:  cached unit outputs (for backprop)
       fn:  an activation function
      vfn:  the numpy vectorization of the activation function
     d_fn:  the derivative of the activation function
    '''

    __slots__ = ('w', 'L', 'k', 'a', 'z', 'fn', 'd_fn', 'vfn')

    def __init__(self, layers, din, dout, mode='regression', fn=activations.sigmoid, d_fn=activations.d_sigmoid):
        self.L = len(layers)
        self.k = [din] + layers + [dout]
        self.fn = activations.sigmoid               
        self.vfn = np.vectorize(self.fn)
        self.d_fn = activations.d_sigmoid

        # Allocate and initialize all the weights.
        # Each layer has +1 for bias.        
        self.w = [None]
        for n in range(len(self.k)-1):
            self.w.append(rand_mat(self.k[n+1], self.k[n]+1))

        
    def eval(self, x):
        self.a = [None for w in range(self.L+2)]   # note a[0] unused
        self.z = [None for w in range(self.L+1)]   # note z[L] is last layer


        # Define variables to make subsequent code cleaner.
        W = self.w
        L = self.L

        self.z[0] = np.append(x,1)
        # Calculate activations and outputs of all layers
        for l in range(1, L+1):
            self.a[l] = np.dot(W[l], self.z[l-1])
            self.z[l] = np.append(self.vfn(self.a[l]), 1)
        # Calculate outputs
        self.a[L+1] = np.dot(W[L+1], self.z[L])

        return self.a[L+1]
        
        
    def fd_grad(self, data, lossfn, delta = 1e-6):
        '''Finite differences approximation to gradient of lossfn.'''
        grad = [None] + [np.zeros(w.shape) for w in self.w[1:]]

        for (x,y) in data:
            cached = lossfn(self.eval(x), y)
            for l in range(1, self.L+2):
                for k in range(self.k[l]):
                    for j in range(self.k[l-1]+1):
                        self.w[l][k][j] += delta
                        update = lossfn(self.eval(x), y)
                        self.w[l][k][j] -= delta
                        grad[l][k][j] += (update-cached)/delta
        N = len(data)
        for n in range(1, len(grad)):
            grad[n] = grad[n] / N
            
        return grad

    
    def bp_grad(self, data, lossfn=sum_squares, dw=1e-6):
        global bptime
        global bpcalls
        bpcalls += 1
        tbegin = time()
        L = self.L
        grad = [None] + [np.zeros(_.shape) for _ in self.w[1:]]
        delta = [None] + [np.zeros(_.shape[0]) for _ in self.w[1:]]
        for (x,y) in data:
            self.eval(x)

            for k in range(self.k[L+1]):
                delta[L+1][k] = diff(lossfn, y, self.a[L+1], k)
                for j in range(self.k[L]+1):
                    grad[L+1][k][j] += delta[L+1][k] * self.z[L][j]
                    
            for n in range(L, 0, -1):
                for k in range(self.k[n]):
                    dsum = 0
                    for i in range(self.k[n+1]):
                        dsum += delta[n+1][i] * self.w[n+1][i][k]
                    delta[n][k] = dsum * self.d_fn(self.a[n][k])

                    for j in range(self.k[n-1]+1):
                        grad[n][k][j] += delta[n][k]*self.z[n-1][j]

        N = len(data)
        if N > 0:
            for l in range(1, L+2):
                for k in range(self.k[l]):
                    for j in range(self.k[l-1]+1):
                        grad[l][k][j] = grad[l][k][j]/N
        bptime += time()-tbegin
        return grad


    def sgd_step(self, data, lossfn, gamma=1, cutoff=1):
        global tgrad
        tbegin = time()
        grad = self.bp_grad([random.choice(data)], lossfn)
        for l in range(1, len(self.w)):
            self.w[l] = self.w[l] - gamma * grad[l]
        tgrad += time() - tbegin


    def gd_step(self, data, lossfn, gamma=1):
        global tgrad
        tbegin = time()
        grad = self.bp_grad(data, lossfn)
        #grad = self.fd_grad(data, lossfn)
        for l in range(1, len(self.w)):
            self.w[l] = self.w[l] - gamma * grad[l]
        tgrad += time() - tbegin



    def loss(self, data, lossfn=sum_squares):
        ''' Return average of lossfn over all data. '''
        result = 0.0
        for (xn, yn) in data:
            result += lossfn(self.eval(xn), yn)
        return result/len(data)

    def save(self,filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

def load_mlp(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)


