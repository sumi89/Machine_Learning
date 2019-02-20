#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:04:21 2018

@author: sumi
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.metrics import f1_score

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def preprocess(dataset, preprocess):
#loading the dataset     
    if(dataset == 1):
        people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
        mask = np.zeros(people.target.shape, dtype=np.bool)
        for target in np.unique(people.target):
            mask[np.where(people.target == target)[0][:50]] = 1    
        X_people = people.data[mask]
        y_people = people.target[mask]
        X_people = X_people / 255.
        # splitting the labelled face dataset
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=16)
        
    elif(dataset == 2):
        data = np.loadtxt('/Users/sumi/python/mnist.txt',delimiter=',')
        train_set, test_set = train_test_split(data,test_size=0.20,random_state = 16)
        X_train = train_set[:,:train_set.shape[1]-1]
        y_train = train_set[:,train_set.shape[1]-1]
        X_test = test_set[:,:test_set.shape[1]-1]
        y_test = test_set[:,test_set.shape[1]-1]
        
#preprocessing the features of test and training data

    if(preprocess == 0): #no preprocessing
        print("No preprocessing:");
        X_train_proc = X_train;
        X_test_proc = X_test;
        return X_train_proc, X_test_proc, y_train, y_test;
    
    elif(preprocess == 2): # normalization
        norm = preprocessing.Normalizer().fit(X_train);
        X_train_proc = norm.transform(X_train);
        X_test_proc =norm.transform(X_test);
        print("Preprocessing: Normalization");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 1): # standarization
        sc = preprocessing.StandardScaler().fit(X_train);
        X_train_proc = sc.transform(X_train);
        X_test_proc = sc.transform(X_test);
        print("Preprocessing: Standarization");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 3): # PCA
        pca = PCA(n_components = 150,random_state = 16).fit(X_train);
        X_train_proc = pca.transform(X_train);
        X_test_proc = pca.transform(X_test);
        print("Preprocessing: PCA");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 4): # nmf
        nmf = NMF(n_components = 150, random_state = 16).fit(X_train);
        X_train_proc = nmf.transform(X_train);
        X_test_proc = nmf.transform(X_test);
        print("Preprocessing: NMF");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 5): # kmeans
        # k-means clustering
        kmeans = KMeans(n_clusters = 150);
        kmeans.fit(X_train);
        X_train_proc = kmeans.transform(X_train);
        X_test_proc = kmeans.transform(X_test);
        print("Preprocessing: k-means");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 7): # standardization and PCA
        sc = preprocessing.StandardScaler().fit(X_train);
        X_train_sc = sc.transform(X_train);
        X_test_sc = sc.transform(X_test);
        pca = PCA(n_components = 150,random_state = 16).fit(X_train_sc);
        X_train_proc = pca.transform(X_train_sc);
        X_test_proc = pca.transform(X_test_sc);
        print("Preprocessing: Standardization and PCA");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 6): # normalization and PCA
        norm = preprocessing.Normalizer().fit(X_train);
        X_train_norm = norm.transform(X_train);
        X_test_norm =norm.transform(X_test);
        pca = PCA(n_components = 150).fit(X_train_norm);
        X_train_proc = pca.transform(X_train_norm);
        X_test_proc = pca.transform(X_test_norm);
        print("Preprocessing: Normalization and PCA");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 9): # standardization and NMF
        sc = preprocessing.StandardScaler().fit(X_train);
        X_train_sc = sc.transform(X_train);
        X_test_sc = sc.transform(X_test);
        nmf = NMF(n_components = 2, random_state = 16).fit(X_train_sc);
        X_train_proc = nmf.transform(X_train_sc);
        X_test_proc = nmf.transform(X_test_sc);
        print("Preprocessing: Standardization and NMF");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 8):# normalization and NMF
        norm = preprocessing.Normalizer().fit(X_train);
        X_train_norm = norm.transform(X_train);
        X_test_norm =norm.transform(X_test);
        nmf = NMF(n_components = 150, random_state = 16).fit(X_train_norm);
        X_train_proc = nmf.transform(X_train_norm);
        X_test_proc = nmf.transform(X_test_norm);
        print("Preprocessing: Normalization and NMF");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 11): # standardization and k-means
        sc = preprocessing.StandardScaler().fit(X_train);
        X_train_sc = sc.transform(X_train);
        X_test_sc = sc.transform(X_test);
        kmeans = KMeans(n_clusters = 150);
        kmeans.fit(X_train_sc);
        X_train_proc = kmeans.transform(X_train_sc);
        X_test_proc = kmeans.transform(X_test_sc);
        print("Preprocessing: Standardization and k-means");
        return X_train_proc, X_test_proc, y_train, y_test;
    elif(preprocess == 10): # normalization and k-means
        norm = preprocessing.Normalizer().fit(X_train);
        X_train_norm = norm.transform(X_train);
        X_test_norm =norm.transform(X_test);
        kmeans = KMeans(n_clusters = 150);
        kmeans.fit(X_train_norm);
        X_train_proc = kmeans.transform(X_train_norm);
        X_test_proc = kmeans.transform(X_test_norm);
        print("Preprocessing: Normalization and k-means");
        return X_train_proc, X_test_proc, y_train, y_test;
        
        
        
        
X_train_proc, X_test_proc, y_train, y_test = preprocess(2,1)
        
def classification_alg(algorithm):
    if(algorithm == 1):
        classifier = KNeighborsClassifier(n_neighbors=10);
        print("Classification: k-nearest neighbors");
        return classifier;
    elif(algorithm == 2):
        classifier = tree.DecisionTreeClassifier(max_depth = 10)
        print("Classification: Decision trees")
        return classifier 
    elif(algorithm == 3):
        classifier = RandomForestClassifier(n_estimators = 200)
        print("Classification: Random forests")
        return classifier 
    elif(algorithm == 4):
        classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
        print("Classification: Support vector machine")
        return classifier 
    elif(algorithm == 5):
        classifier =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 200))
        print("Classification: Multilayer perceptron")
        return classifier 
    
classifier = classification_alg(4)
classifier.fit(X_train_proc, y_train)
pred = classifier.predict(X_test_proc)
# scoring on the scaled test set
print("Test set accuracy: {:.3f}".format(classifier.score(X_test_proc, y_test)))
# scoring on the scaled train set
print("Train set accuracy:: {:.3f}".format(classifier.score(X_train_proc, y_train)))

print("f1:: {:.3f}".format(f1_score(y_test,pred)))


       
        
        


#  knn algo
#knn = KNeighborsRegressor(n_neighbors=4, weights = 'distance')
#knn.fit(x_train, y_train)
#y_pred_knn = knn.predict(x_test)
#print("mse for k-nearest neighbor {}".format(mean_squared_error(y_pred_knn,y_test)))
#print("test accuracy for k-nearest neighbor {}".format(knn.score(x_test, y_test)))



