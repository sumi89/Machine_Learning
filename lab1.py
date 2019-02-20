# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def read_data(filename):
    D = np.loadtxt(filename, delimiter=",")
    # or we can do this "np.random.shuffle(D)"
    # print(D.shape)
    X = D[:,:4] # it will take col 0,1,2,3
    # X = D[:,0:4] or D[4:150,:4]
    Y = D[:,4].astype(int)
    return X,Y

def split(X,Y,f):
    last = np.round(X.shape[0]*f).astype(int) # ??? can't understand f
    rp = np.random.permutation(X.shape[0])  # shuffling rows
    X_train = X[rp[:last]]  # rows from beginning through last-1
    X_test = X[rp[last:]]   # rows from last through the rest of the array
    Y_train = Y[rp[:last]]
    Y_test = Y[rp[last:]]
    return X_train, Y_train, X_test, Y_test

def majority(X):    # didn't get the big picture ?????? - which class has more samples
    mx = np.max(X)  # input is the Y_train set
    count = np.zeros(mx+1,).astype(int) 
    for i in range(mx+1):
        count[i] = np.sum(X==i)
    print(count)
    return np.argmax(count)

def accuracy(L1,L2):
    return np.sum(L1==L2)/L2.shape[0]

def oneNN(Xtr,Ytr,Xts):
    #Xts = Xts.reshape(1,-1) # for 1 row of test set
    pred = np.zeros(Xts.shape[0],).astype(int)
    for i in range(Xts.shape[0]):
        d = np.sum(np.square(Xtr-Xts[i]),axis=1)
        pred[i] = Ytr[np.argmin(d)] # argmin gives the position of min
    print("pred: {}".format(pred))    
    return pred

Features, Label = read_data("iris_numlabel.txt")
Features_tr, Label_tr, Features_ts, Label_ts = split(Features, Label, .8)
print(majority(Label_tr))
print("Accuracy majority: {}".format(accuracy(majority(Label_tr),Label_ts)))
#print("Accuracy 1-NN: {}".format(accuracy(Label_ts,oneNN(Features_tr,Label_tr,Features_ts[0,:])))) #for 1 row of test set
print("Accuracy 1-NN: {}".format(accuracy(Label_ts,oneNN(Features_tr,Label_tr,Features_ts))))