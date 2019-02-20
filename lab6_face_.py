#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 02:17:45 2018

@author: sumi
"""



from __future__ import print_function


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
import cv2

from sklearn.datasets import fetch_lfw_people


def onehot(X):
    T = np.zeros((X.shape[0],np.max(X)+1))
    T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
    return T

def confusion_matrix(Actual,Pred):
    cm=np.zeros((np.max(Actual)+1,np.max(Actual)+1), dtype=np.int)
    for i in range(len(Actual)):
        cm[Actual[i],Pred[i]]+=1
    return cm

def read_data(X,Class):
    print("Reading data")
    #X = np.loadtxt(xfile, delimiter=",")
    #Class = np.loadtxt(yfile, delimiter=",").astype(int)
    X /= 255
    X = X.reshape(-1,62,47,1)
    Y = onehot(Class)
    print("Data read")
    return X,Y,Class



people = fetch_lfw_people(min_faces_per_person=50, resize=0.5)



mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1 
    
X_people = people.data[mask]
y_people = people.target[mask]
# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
#X_people = X_people / 255.

#Read data  
# number 1
X_train, X_test, Y_train, Y_test = train_test_split(X_people, y_people, random_state=99)

x_train, y_train, train_class = read_data(X_train, Y_train)
x_test, y_test, test_class = read_data(X_test, Y_test)

# Set network parameters 
np.set_printoptions(threshold=np.inf) #Print complete arrays
batch_size = 32
epochs = 100
learning_rate = 0.0001
first_layer_filters = 16
second_layer_filters = 32
third_layer_filters=32
ks = 3
mp = 2
dense_layer_size = 64


# trial
flipped_image = np.zeros([people.images.shape[0],people.images.shape[1],people.images.shape[2]])

for i in range(people.images.shape[0]):
#    ori_image = people.images[i]
#    plt.imshow(ori_image)
    flipped_image[i] = cv2.flip(people.images[i], 1)
#    plt.imshow(flipped_image)
    
flipped_image = flipped_image.reshape(1560,-1)
new_train_data = np.append(people.data, flipped_image, axis = 0)
new_train_label = np.append(people.target, people.target, axis = 0)




#Read data 
#number 3
X_train, X_test, Y_train, Y_test = train_test_split(new_train_data, new_train_label, random_state=99)

x_train, y_train, train_class = read_data(X_train, Y_train)
x_test, y_test, test_class = read_data(X_test, Y_test)

#
#for i in range(3):
#    ind = np.random.randint(x_train.shape[0])
#    I = (255*x_train[ind]).reshape((65, 87)).astype(int)
#    plt.imshow(I, cmap=plt.get_cmap('gray'))
#    plt.show()

#Build model  
model = Sequential()
model.add(Conv2D(first_layer_filters, kernel_size=(ks, ks),
                 activation='relu',
                 input_shape= (62, 47, 1)))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(mp, mp)))
print(model.output_shape)
model.add(Conv2D(second_layer_filters, (ks, ks), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(mp, mp)))
print(model.output_shape)
model.add(Conv2D(third_layer_filters, (ks, ks), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(mp, mp)))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(dense_layer_size, activation='relu'))
print(model.output_shape)
model.add(Dense(12, activation='softmax'))
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

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred=model.predict_classes(x_test)
print('\n',confusion_matrix(test_class,pred))
print(model.summary())


