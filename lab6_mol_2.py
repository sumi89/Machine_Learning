#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 03:00:45 2018

@author: sumi
"""

from __future__ import print_function
import cv2
import os

import sklearn
from sklearn.model_selection import train_test_split

from scipy.special import expit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize

def onehot(X):
    T = np.zeros((X.shape[0],np.max(X)+1))
    T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
    return T

def confusion_matrix(Actual,Pred):
    cm=np.zeros((np.max(Actual)+1,np.max(Actual)+1), dtype=np.int)
    for i in range(len(Actual)):
        cm[Actual[i],Pred[i]]+=1
    return cm

def load_data(xfile1,xfile2):
    X1 = np.load(xfile1)
    X1_class = np.ones(X1.shape[0],).astype(int)
    X1_class = X1_class.reshape(-1,1)
    
    X2 = np.load(xfile2)
    X2_class = np.zeros(X2.shape[0],).astype(int)
    X2_class = X2_class.reshape(-1,1)
    
    X = np.append(X1,X2, axis = 0)
    Class = np.append(X1_class,X2_class, axis = 0)
    return X, Class



def read_data(X, Class):
    print("Reading data")
    #X /= 255
    X = X.reshape(-1,7,7,1)
    Y = onehot(Class)
    print("Data read")
    return X,Y,Class

def detect_object(I,regressor,m,n):
    best_match=-1
    best_r = -1
    best_c = -1
    for r in range(I.shape[0]-m+1):
        for c in range(I.shape[1]-n+1):
            subIm = normalize(I[r:r+m,c:c+n].reshape(1, -1))
            result= regressor.predict(subIm)
            if result>best_match:
                best_match = result
                best_r = r
                best_c=c
    return best_r, best_c



# Set network parameters 
np.set_printoptions(threshold=np.inf) #Print complete arrays
batch_size = 32
epochs = 10
learning_rate = 0.0001
#first_layer_filters = 8
#second_layer_filters = 16
first_layer_filters = 16
second_layer_filters = 32
ks = 3
mp = 2
dense_layer_size = 64

#Read data  

X, Y = load_data("/Users/sumi/python/molecules_.npy", "/Users/sumi/python/no_mol_.npy")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=99)

x_train, y_train, train_class = read_data(X_train, Y_train)
x_test, y_test, test_class = read_data(X_test, Y_test)



#Build model  
model = Sequential()
model.add(Conv2D(first_layer_filters, kernel_size=(ks, ks),
                 activation='relu',
                 input_shape= (7, 7, 1)))
print(model.output_shape)
#model.add(MaxPooling2D(pool_size=(mp, mp)))
#print(model.output_shape)
model.add(Conv2D(second_layer_filters, (ks, ks), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(mp, mp)))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(dense_layer_size, activation='relu'))
print(model.output_shape)
model.add(Dense(2, activation='softmax'))
print(model.output_shape)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
print(model.output_shape)

#Train network, store history in history variable
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])



# trial
img=mpimg.imread('/Users/sumi/python/mol_imgs/00020.tif')
#plt.subplot(1, 3, 1)
imgplot = plt.imshow(img,'gray')
I = np.float32(img)
#I = I.reshape(-1,64,64,1)
m = 7
n = 7

subIm = np.zeros([m,n])
subIm = subIm.reshape(-1,m,n,1)

pred_prob = np.zeros([img.shape[0]- m+1, img.shape[1]- n+1])

for r in range(I.shape[0]-m+1):
    for c in range(I.shape[1]-n+1):
        subIm = normalize(I[r:r+m,c:c+n]).reshape(1,-1)
        result= model.predict(subIm.reshape(-1,m,n,1))
        pred_prob[c][r] = result[:,1]
best_r,best_c = np.unravel_index(np.argmax(pred_prob, axis=None), pred_prob.shape)
      
# to get best prob
#best = 0
#for i in range(I.shape[0]-m+1):
#    for j in range(I.shape[0]-n+1):
#        if(pred_prob[i][j] > best):
#            best = pred_prob[i][j]
#            best_r = i
#            best_c = j
#        
#best_r,best_c = np.unravel_index(np.argmax(pred_prob, axis=None), pred_prob.shape)
#plt.subplot(1, 3, 2)
plt.imshow(pred_prob,'gray')

iii = cv2.rectangle(I, (4,19), (10,25), -1)
alpha = 0.1
iii = cv2.addWeighted(I,alpha, I, 1-alpha , 0, I )
#plt.subplot(1, 3, 3)
plt.imshow(iii,'gray')


plt.subplot(1, 3, 1)
imgplot = plt.imshow(img,'gray')
plt.subplot(1, 3, 2)
plt.imshow(pred_prob,'gray')
plt.subplot(1, 3, 3)
plt.imshow(iii,'gray')
#

###############################
"IT WORKS"

import PIL
from PIL import Image
import glob

from matplotlib import pyplot as plt



image_list = []
for filename in glob.glob('/Users/sumi/python/ml/mol_imgs/*.tif'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)

    
for img in image_list:
    # you can show every image
    img.show()
#####################################






import PIL
from PIL import Image
#import glob
#
#for imge in glob.glob("*.tif"):
#    img = Image.open(imge)
#    img.show()
#

import PIL.Image
direc = '/Users/sumi/python/ml/mol_imgs'
IMAG = os.listdir(direc)
for imge in IMAG:
    img = Image.open(imge)
    
    #img=mpimg.imread(imge)
    imgplot = plt.imshow(img,'gray')
    I = np.float32(img)
    
import os, shutil

train_dir = os.path.join(direc, 'train')
os.mkdir(train_dir)


fnames = ['{}.tif'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(direc, fname)
    dst = os.path.join(train_dir, fname)
    shutil.copyfile(src, dst)
    

