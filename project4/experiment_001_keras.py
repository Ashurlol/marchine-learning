import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adagrad
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD
from time import time
import numpy as np
import matplotlib.pyplot as plt

records = {}
xList, y1List, y2List = [], [], []

def load_sonar(path):
    xdata = []
    ydata = []
    with open(path) as fh:
        data = []
        for line in fh:
            elt = line.strip().split(',')
            xvals = [float(x) for x in elt[:-1]]

            xdata.append(xvals)
            
            yval = int(elt[-1])

            if yval == 0:
                ydata.append([1, 0])
            elif yval == 1:
                ydata.append([0, 1])
            else:
                print('unexpected yval=',yval)
    return np.array(xdata), np.array(ydata)

def expt_001():
    xtrain , ytrain = load_sonar('data/sonar.train')
    xdev , ydev = load_sonar('data/sonar.dev')

    gamma = 1
    epochs = 10000
    
    layers = [12]  # number of hidden units

    savefreq = 0   # store the model every savefreq steps
    logfreq = 100    # print updates every logfreq steps
    

    din = len(xtrain[0])
    dout = len(ytrain[0])

    model = Sequential()
    if len(layers) == 0:
        model.add(Dense(dout, input_shape=(din,)))
    else:
        model.add(Dense(layers[0], input_shape=(din,), activation='sigmoid'))
        for h in list(layers[1:]):
            model.add(Dense(h))
        model.add(Dense(dout))
    
    model.summary()
    model.compile(loss='mean_squared_error',
                  #optimizer=Adam(),
                  optimizer=SGD(lr=gamma),
                  metrics=['accuracy'])


    for epoch in range(epochs):
        history = model.fit(xtrain, ytrain,
                            epochs=(epoch+1),
                            initial_epoch=epoch,
                            batch_size=len(xtrain),
                            verbose=0,
                            validation_data=(xdev, ydev))

        counter = epoch+1
        
        score = model.evaluate(xtrain, ytrain, verbose=0)
        #print('Train loss:', score[0])
        #print('Train accuracy:', score[1])
        tacc = score[1]
        score = model.evaluate(xdev, ydev, verbose=0)
        #print('Dev loss:', score[0])
        #print('Dev accuracy:', score[1])
        dacc = score[1]

        records[counter] = records.get(counter, {})
        records[counter]["train"] = records[counter].get("train",[])+[tacc]
        records[counter]["dev"] = records[counter].get("dev",[])+[dacc]

        print('Accuracy after %d rounds is %f train, %f dev '%(epoch, tacc, dacc))

def average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    for i in range(30):
        expt_001()
        print(i+1)
    iteration = sorted(records.items())
    for stuff in iteration:
        xList.append(stuff[0])
        y1List.append(average(stuff[1]["train"]))
        y2List.append(average(stuff[1]["dev"]))



    fig1, ax1 = plt.subplots()
    ax1.set_title('Mean Accuracy vs Epochs (MLP)')    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accurancy')
    ax1.plot(xList, y1List, 'b', label='training')
    ax1.plot(xList, y2List, 'g', label='development')
    ax1.legend()
    ax1.figure.savefig('part2-2.png')

    fig2, ax2 = plt.subplots()
    ax2.set_title('Box-and-whiskers Plot (MLP)')
    ax2.boxplot([y1List,y2List])
    ax2.set_xticklabels(["training", "development"])
    ax2.figure.savefig('part2-3.png')