import matplotlib.pyplot as plt
import random
import math
import numpy as np
import sys

records = {}
xList, y1List, y2List = [], [], []

def load_data(filename):
    # Assume data is in CSV format, float-valued features
    # binary category (zero or one) on right.
    data = np.array([[float(x) for x in line.strip().split(',')]
                     for line in open(filename).readlines()])
    return data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def infer(obs, params):
    """ Calculate sigmoid(w.x) """ 
    acc = params[len(obs)] # bias param in last position
    for n in range(len(obs)-1):
        acc += params[n]*obs[n]
    return sigmoid(acc)
    
def classify(obs, params):
    raw = infer(obs, params)
    return int(raw > 0.5)


def accuracy(data, params):

    correct = 0
    for obs in data:
        if (classify(obs[:-1], params) == obs[-1]):
            correct += 1
    return correct / len(data)

def loss_ss(data, params):
    acc = 0
    for obs in data:
        acc += (infer(obs[:-1],params) - obs[-1])**2
    return acc/len(data)
    
def learn(train, test, steps=1000):
    delta = 1e-3
    gamma = 1    #learning rate
    batch_size = len(train)
    loss = loss_ss
    #loss = accuracy

    N = len(train[0])-1  # Dimension of input (i.e., number of features per example.)
    params = 0.5-np.random.rand(N+1)  # (uniform) random init params in range +- 0.5 centered at zero. 
    np.set_printoptions(linewidth=100, precision=3)
    steps_per_epoch = int(len(train)/batch_size)
    bestTest = accuracy(test, params)
    bestTrain = accuracy(train, params)
    counter = 1
    for epoch in range(steps):
        for i in range(steps_per_epoch):
            sample = random.sample(list(train),batch_size)  # without replacement
            cached = loss(sample, params)
            grad = np.zeros(N+1)
            for n in range(N+1):
                params[n] += delta
                grad[n] = (loss(sample, params)-cached)/delta
                params[n] -= delta
            #print("grad = " + str(grad))
            #params -= gamma**(epoch//10) * grad
            params -= gamma * grad
        
        tacc, dacc = accuracy(train,params), accuracy(test,params)
        records[counter] = records.get(counter, {})
        records[counter]["train"] = records[counter].get("train",[])+[tacc]
        records[counter]["dev"] = records[counter].get("dev",[])+[dacc]
        #xList.append(counter)
        #y1List.append(tacc)
        #y2List.append(dacc)
        counter += 1          
        #print('trainAcc=%.5f\ttestAcc=%.5f'%(tacc, dacc))
    return params

def demo():
    import pdb
    train = load_data(sys.argv[1])
    dev = load_data(sys.argv[2])
    params = learn(train, dev, 1500)
    # (uniform) random init params in range +- 0.5 centered at zero. 
    #params = 0.5-np.random.rand(len(data[0]))  
    print('Final Result: trainAcc=%.5f\ttestAcc=%.5f'%(accuracy(train,params), accuracy(dev,params)))
    
def average(lst):
    return sum(lst) / len(lst)

#xdev , ydev = load_sonar('data/sonar.test')

if __name__ == '__main__':
    for i in range(30):
        demo()
        print(i+1)
    iteration = sorted(records.items())
    for stuff in iteration:
        xList.append(stuff[0])
        y1List.append(average(stuff[1]["train"]))
        y2List.append(average(stuff[1]["dev"]))



    #fig1, ax1 = plt.subplots()
    #ax1.set_title('Accuracy vs Epochs ')
    #ax1.set_xlabel('Epoch')
    #ax1.set_ylabel('Accurancy')
    #ax1.plot(xList, y1List, label='train')
    #ax1.plot(xList, y2List, label='dev')
    #ax1.legend(loc = 'lower right')
    #ax1.figure.savefig('part1-2.png')

    #fig2, ax2 = plt.subplots()
    #ax2.set_title('Box-and-whiskers Plot (Logistic Regression)')
    #ax2.boxplot([y1List,y2List])
    #ax2.set_xticklabels(["training", "development"])
    #ax2.figure.savefig('part1-3.png')

    print('mean = ', average(y2List), 'min = ', min(y2List), 'max = ', max(y2List),
          'standard deviation = ', np.std(y2List))

    fig1, ax1 = plt.subplots()
    ax1.set_title('Mean Accuracy vs Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(xList, y1List, label='train')
    ax1.plot(xList, y2List, label='dev')
    ax1.legend(loc='lower right')
    ax1.figure.savefig('part2-222.png')

    fig2, ax2 = plt.subplots()
    ax2.set_title('Histogram of test set accuracy (Logistic Regression)')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Counts')
    counts, bins = np.histogram(y2List)
    ax2.hist(bins[:-1], bins, weights=counts)
    ax2.figure.savefig('part2-333.png')

