#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 01:17:15 2018

@author: sumi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 23:20:29 2018

@author: sumi
"""

import numpy as np
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


molecules = np.load('/Users/sumi/python/ml/molecules.npy') 
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

"""-----------------------------------------------------------------------------"""

"""MinMaxScaler pre-processing with SVC algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
param_grid = {'svm__C' : [ 1e-4, 0.1],
              'svm__gamma' : [1e-4, 0.1]}

grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("MinMaxScaler pre-processing with SVC algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("Best parameters: {}".format(grid.best_params_))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'svm__C' , 
                           xticklabels = param_grid [ 'svm__C' ],
                           ylabel = 'svm__gamma' , 
                           yticklabels = param_grid [ 'svm__gamma' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""StandardScaler pre-processing with SVC algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
param_grid = {'svm__C' : [1e-6, 0.1],
              'svm__gamma' : [1e-6, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("StandardScaler pre-processing with SVC algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'svm__C' , 
                           xticklabels = param_grid [ 'svm__C' ],
                           ylabel = 'svm__gamma' , 
                           yticklabels = param_grid [ 'svm__gamma' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""


"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""PCA pre-processing with SVC algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", PCA()), ("svm", SVC())])
param_grid = { 'scaler__n_components' : [5],
         'svm__C' : [1e-6, 0.1],
         'svm__gamma' : [1e-6, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train)
pred = grid.predict(X_test)
print("PCA pre-processing with SVC algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'svm__C' , 
                           xticklabels = param_grid [ 'svm__C' ],
                           ylabel = 'svm__gamma' , 
                           yticklabels = param_grid [ 'svm__gamma' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""NMF pre-processing with SVC algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", NMF()), ("svm", SVC())])
param_grid = { 'scaler__n_components' : [5],
         'svm__C' : [0.00001, 0.1],
              'svm__gamma' : [0.00001, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("NMF pre-processing with SVC algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'svm__C' , 
                           xticklabels = param_grid [ 'svm__C' ],
                           ylabel = 'svm__gamma' , 
                           yticklabels = param_grid [ 'svm__gamma' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""


"""==========================================================================================="""
"""==========================================================================================="""


"""-----------------------------------------------------------------------------"""

"""MinMaxScaler pre-processing with knn algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", MinMaxScaler()), ("knn", KNeighborsClassifier())])
param_grid = {  'knn__n_neighbors' : [ 10, 20],
        'knn__weights' : ['uniform', 'distance']}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("MinMaxScaler pre-processing with knn algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'knn__n_neighbors' , 
                           xticklabels = param_grid [ 'knn__n_neighbors' ],
                           ylabel = 'knn__weights' , 
                           yticklabels = param_grid [ 'knn__weights' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""StandardScaler pre-processing with knn algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
param_grid = { 'knn__n_neighbors' : [ 10, 20],
        'knn__weights' : ['uniform', 'distance']}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("StandardScaler pre-processing with knn algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'knn__n_neighbors' , 
                           xticklabels = param_grid [ 'knn__n_neighbors' ],
                           ylabel = 'knn__weights' , 
                           yticklabels = param_grid [ 'knn__weights' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""


"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""PCA pre-processing with knn algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", PCA()), ("knn", KNeighborsClassifier())])
param_grid = { 'scaler__n_components' : [20],
        'knn__n_neighbors' : [ 10, 20],
        'knn__weights' : ['uniform', 'distance']}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("PCA pre-processing with knn algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'knn__n_neighbors' , 
                           xticklabels = param_grid [ 'knn__n_neighbors' ],
                           ylabel = 'knn__weights' , 
                           yticklabels = param_grid [ 'knn__weights' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""NMF pre-processing with knn algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", NMF()), ("knn", KNeighborsClassifier())])
param_grid = { 'scaler__n_components' : [10],
        'knn__n_neighbors' : [ 10, 20],
        'knn__weights' : ['uniform', 'distance']}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("NMF pre-processing with knn algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'knn__n_neighbors' , 
                           xticklabels = param_grid [ 'knn__n_neighbors' ],
                           ylabel = 'knn__weights' , 
                           yticklabels = param_grid [ 'knn__weights' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""
"""==========================================================================================="""


"""-----------------------------------------------------------------------------"""

"""MinMaxScaler pre-processing with Decision tree algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", MinMaxScaler()), ("cls", tree.DecisionTreeClassifier())])
param_grid = { 'cls__max_depth' : [2, 3],
              'cls__max_leaf_nodes' : [10, 16]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("MinMaxScaler pre-processing with Decision tree algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 ) # reshape(y,x)
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'cls__max_depth' , 
                           xticklabels = param_grid [ 'cls__max_depth' ],
                           ylabel = 'cls__max_leaf_nodes' , 
                           yticklabels = param_grid [ 'cls__max_leaf_nodes' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""StandardScaler pre-processing with Decision tree algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", StandardScaler()), ("cls", tree.DecisionTreeClassifier())])
param_grid = { 'cls__max_depth' : [3, 7],
              'cls__max_leaf_nodes' : [10, 16]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("StandardScaler pre-processing with Decision tree algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 ) # reshape(y,x)
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'cls__max_depth' , 
                           xticklabels = param_grid [ 'cls__max_depth' ],
                           ylabel = 'cls__max_leaf_nodes' , 
                           yticklabels = param_grid [ 'cls__max_leaf_nodes' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""


"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""PCA pre-processing with Decision tree algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", PCA()), ("cls", tree.DecisionTreeClassifier())])
param_grid = { 'scaler__n_components' : [10],
        'cls__max_depth' : [5, 7],
        'cls__max_leaf_nodes' : [10, 16]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("PCA pre-processing with Decision tree algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'cls__max_depth' , 
                           xticklabels = param_grid [ 'cls__max_depth' ],
                           ylabel = 'cls__max_leaf_nodes' , 
                           yticklabels = param_grid [ 'cls__max_leaf_nodes' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""NMF pre-processing with Decision tree algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", NMF()), ("tree", DecisionTreeClassifier())])
param_grid = { 'scaler__n_components' : [5],
        'tree__max_depth' : [5, 7],
        'tree__max_leaf_nodes' : [10, 16]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("NMF pre-processing with Decision tree algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'tree__max_depth' , 
                           xticklabels = param_grid [ 'tree__max_depth' ],
                           ylabel = 'tree__max_leaf_nodes' , 
                           yticklabels = param_grid [ 'tree__max_leaf_nodes' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""
"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""MinMaxScaler pre-processing with Random forest algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", MinMaxScaler()), ("forest", RandomForestClassifier())])
param_grid = { 'forest__n_estimators' : [ 15, 20],
              'forest__max_depth' : [3, 5]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("MinMaxScaler pre-processing with Random forest algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'forest__n_estimators' , 
                           xticklabels = param_grid [ 'forest__n_estimators' ],
                           ylabel = 'forest__max_depth' , 
                           yticklabels = param_grid [ 'forest__max_depth' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""StandardScaler pre-processing with Random forest algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", StandardScaler()), ("forest", RandomForestClassifier())])
param_grid = { 'forest__n_estimators' : [ 15, 30],
              'forest__max_depth' : [3, 7]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("StandardScaler pre-processing with Random forest algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'forest__n_estimators' , 
                           xticklabels = param_grid [ 'forest__n_estimators' ],
                           ylabel = 'forest__max_depth' , 
                           yticklabels = param_grid [ 'forest__max_depth' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""


"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""PCA pre-processing with Random forest algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", PCA()), ("forest", RandomForestClassifier())])
param_grid = { 'scaler__n_components' : [5],
        'forest__n_estimators' : [15, 30],
              'forest__max_depth' : [5, 10]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("PCA pre-processing with Random forest algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'forest__n_estimators' , 
                           xticklabels = param_grid [ 'forest__n_estimators' ],
                           ylabel = 'forest__max_depth' , 
                           yticklabels = param_grid [ 'forest__max_depth' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""NMF pre-processing with Random forest algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", NMF()), ("forest", RandomForestClassifier())])
param_grid = { 'scaler__n_components' : [5],
        'forest__n_estimators' : [ 15, 30],
              'forest__max_depth' : [ 5, 15]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("NMF pre-processing with Random forest algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'forest__n_estimators' , 
                           xticklabels = param_grid [ 'forest__n_estimators' ],
                           ylabel = 'forest__max_depth' , 
                           yticklabels = param_grid [ 'forest__max_depth' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""
"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""MinMaxScaler pre-processing with Multilayer perceptron algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", MinMaxScaler()), ("mlp", MLPClassifier())])
param_grid = {'mlp__hidden_layer_sizes': [(10,10) , (20,20)],
        'mlp__alpha': [1e-5, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("MinMaxScaler pre-processing with Multilayer perceptron algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'mlp__hidden_layer_sizes' , 
                           xticklabels = param_grid [ 'mlp__hidden_layer_sizes' ],
                           ylabel = 'mlp__alpha' , 
                           yticklabels = param_grid [ 'mlp__alpha' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""StandardScaler pre-processing with Multilayer perceptron algorithm"""
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier())])
param_grid = {'mlp__hidden_layer_sizes': [(10,10), (20,20)],
        'mlp__alpha': [1e-5, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("StandardScaler pre-processing with Multilayer perceptron algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'mlp__hidden_layer_sizes' , 
                           xticklabels = param_grid [ 'mlp__hidden_layer_sizes' ],
                           ylabel = 'mlp__alpha' , 
                           yticklabels = param_grid [ 'mlp__alpha' ], cmap = "viridis" )


"""-----------------------------------------------------------------------------"""


"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""PCA pre-processing with Multilayer perceptron algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", PCA()), ("mlp", MLPClassifier())])
param_grid = { 'scaler__n_components' : [5],
        'mlp__hidden_layer_sizes': [(10,10),(20,20)],
        'mlp__alpha': [1e-5, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("PCA pre-processing with Multilayer perceptron algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'mlp__hidden_layer_sizes' , 
                           xticklabels = param_grid [ 'mlp__hidden_layer_sizes' ],
                           ylabel = 'mlp__alpha' , 
                           yticklabels = param_grid [ 'mlp__alpha' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""

"""==========================================================================================="""

"""-----------------------------------------------------------------------------"""

"""NMF pre-processing with Multilayer perceptron algorithm """
##Pipelines in Grid Searches
pipe = Pipeline([("scaler", NMF()), ("mlp", MLPClassifier())])
param_grid = { 'scaler__n_components' : [5],
        'mlp__hidden_layer_sizes': [(10,10),(20,20)],
        'mlp__alpha': [1e-5, 0.1]}
grid = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
grid.fit(X_train,y_train )
pred = grid.predict(X_test)
print("NMF pre-processing with Multilayer perceptron algorithm")
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set accuracy: {:.2f}".format(grid.score(X_test,y_test)))
print("f1 score: {:.2f}".format(f1_score(y_test,pred)))
print("Best parameters: {}".format(grid.best_params_))
print ( classification_report ( y_test, pred, target_names = [ "mol" , "no_mol" ]))
scores = grid.cv_results_ [ 'mean_test_score' ] . reshape ( 2,2 )
# plot the mean cross-validation scores
mglearn . tools . heatmap ( scores , xlabel = 'mlp__hidden_layer_sizes' , 
                           xticklabels = param_grid [ 'mlp__hidden_layer_sizes' ],
                           ylabel = 'mlp__alpha' , 
                           yticklabels = param_grid [ 'mlp__alpha' ], cmap = "viridis" )

"""-----------------------------------------------------------------------------"""
