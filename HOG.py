#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:48:34 2018

@author: sumi
"""


import numpy as np


from sklearn.model_selection import train_test_split


molecules = np.load('/Users/sumi/python/molecules.npy') 
molecules =  molecules.reshape(molecules.shape[0],-1)
postive_target = np.ones(molecules.shape[0],).astype(int)
postive_target = postive_target.reshape(-1,1)
#mol_pos_tar = np.append(molecules, postive_target, axis = 1)

no_molecules = np.load('/Users/sumi/python/no_mol.npy') 
#no_molecules = np.load('D:\spring 2018\ML\labs\lab4\no_mol.npy')
no_molecules =  no_molecules.reshape(no_molecules.shape[0],-1)
negative_target = np.zeros(no_molecules.shape[0],).astype(int)
negative_target = negative_target.reshape(-1,1)

#no_mol_neg_tar = np.append(no_molecules, negative_target, axis = 1)

features_mol = np.append(molecules,no_molecules, axis = 0)
target_mol = np.append(postive_target,negative_target, axis = 0)

X_train, X_test, y_train, y_test = train_test_split(features_mol, target_mol, random_state = 16)

#Vg = np.zeros(7)
#Hg = np.zeros(7)
#Mg = np.zeros(target_mol.shape)
#I = no_molecules
#
#for i in range(7):
#    for j in range(7):
#        Vg[i][j] = I[i+1][j] - I[i][j]
#        Hg[i][j] = I[i][j+1] - I[i][j]
#    
#        Mg[i][j] = np.sqrt(np.square(Vg[i][j])+np.square(Hg[i][j]))
        
        
def magnitudes(X):
    X_hor_top = X[:, :]
    X_hor_bottom = X[:, :]
    h_g = np.subtract(X_hor_top, X_hor_bottom) 
    # concatenate row of zero at the bottom
    X_vert_left = X[:-1, :]
    X_vert_right = X[1:, :]
    v_g = np.subtract[X_vert_right, X_vert_left]
    # concatenate col of zero on the right
    return h_g, v_g

I = np.matrix([[1, 5, 2], [3, 6, 4], [2, 1, 6]])
I = np.matrix('1 5 2 ; 3 6 4 ; 2 1 6')
result = magnitudes(I)
print(result)
       
        
        
        