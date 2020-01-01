from time import time
import activations
import numpy as np
import mlp
import loss
import matplotlib.pyplot as plt

records = {}
xList, y1List, y2List = [], [], []

def load_sonar(path):
    with open(path) as fh:
        data = []
        for line in fh:
            elt = line.strip().split(',')
            xvals = [float(x) for x in elt[:-1]]
            yval = int(elt[-1])

            if yval == 0:
                data.append((xvals, [1, 0]))
            elif yval == 1:
                data.append((xvals, [0, 1]))
            else:
                print('unexpected yval=',yval)
    return data

def expt_001():
    train = load_sonar('data/sonar.train')
    dev = load_sonar('data/sonar.dev')
    test = load_sonar('data/sonar.test')


    gamma = 1     # the learning rate
    epochs = 100  # the maximum number of epochs
    


    layers = [12]  # one hidden layer with one hidden unit
    din = len(train[0][0])
    dout = len(train[0][1])

    m = mlp.MLP(layers, din, dout)


    lossfn = loss.sum_squares


    savefreq = 0   # store the model every savefreq steps
    logfreq = 1    # print updates every logfreq steps
    step = 0
    tsum = 0
    tbegin = time()
    for epoch in range(epochs):
        if ((epoch % logfreq) == 0):
            tloss = time()
            lss_train = m.loss(train, loss.zeroone)
            lss_dev = m.loss(dev, loss.zeroone)                
            tloss = time() - tloss
            
            counter, tacc, dacc = epoch+1, 1-lss_train, 1-lss_dev
            records[counter] = records.get(counter, {})
            records[counter]["train"] = records[counter].get("train",[])+[tacc]
            records[counter]["dev"] = records[counter].get("dev",[])+[dacc]

            print('Accuracy after %d rounds is %f train, %f dev '%(epoch, tacc, dacc))
        m.gd_step(train, lossfn, gamma=gamma)

def average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    for i in range(1):
        expt_001()
        print(i+1)
    iteration = sorted(records.items())
    for stuff in iteration:
        xList.append(stuff[0])
        y1List.append(average(stuff[1]["train"]))
        y2List.append(average(stuff[1]["dev"]))



    fig1, ax1 = plt.subplots()
    ax1.set_title('Mean Accuracy vs Epochs (Logistic Regression)')    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accurancy')
    ax1.plot(xList, y1List, 'b', label='training')
    ax1.plot(xList, y2List, 'g', label='development')
    ax1.legend()
    ax1.figure.savefig('part2-1.png')

    fig2, ax2 = plt.subplots()
    ax2.set_title('Box-and-whiskers Plot (Logistic Regression)')
    ax2.boxplot([y1List,y2List])
    ax2.set_xticklabels(["training", "development"])
    ax2.figure.savefig('part2-3.png')