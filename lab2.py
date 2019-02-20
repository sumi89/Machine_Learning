#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 00:41:10 2018

@author: sumi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:42:04 2018

@author: sumi
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/sumi/python/energy.txt',delimiter=',')
train_set, test_set = train_test_split(data,test_size=0.20,random_state = 67)
x_train = train_set[:,:train_set.shape[1]-1]
y_train = train_set[:,train_set.shape[1]-1]
x_test = test_set[:,:test_set.shape[1]-1]
y_test = test_set[:,test_set.shape[1]-1]

#plt.plot(Features[:0]) # plots 1st feature


#
#regression_knn = KNeighborsRegressor(n_neighbors=20)
#regression_lr = linear_model.LinearRegression()
#regression_rr = linear_model.Ridge(alpha = 10)
regression_rt = tree.DecisionTreeRegressor(max_depth=8)
#regression_rf = RandomForestRegressor(n_estimators = 10000,random_state = 16)

#################
## compute the minimum value per feature on the training set
#min_on_training_svr = x_train.min(axis=0)
### compute the range of each feature (max - min) on the training set
#range_on_training_svr = (x_train - min_on_training_svr).max(axis=0)
#
### subtract the min, and divide by range 
### afterward, min=0 and max=1 for each feature
#x_train_scaled_svr = (x_train - min_on_training_svr)/range_on_training_svr
### use the same transformation on the test set,
### using min and range of the training set
#x_test_scaled_svr = (x_train - min_on_training_svr)/range_on_training_svr
########################
#regression_svr = svm.SVR()
#regression_svr = svm.SVR(C=0.1,gamma = 0.1)
#regression_svr = svm.SVR(kernel = 'linear', C = 0.1, gamma = 0.1)

##########################
## compute mean value per feature on the training set
#mean_on_train_mlp=x_train.mean(axis=0)
## compute the standard deviation of each feature on the training set
#std_on_train_mlp=x_train.std(axis=0)
#
## subtract the mean, and scale by inverse standard deviation afterward,mean=0,std=1
#x_train_scaled_mlp = (x_train - mean_on_train_mlp)/std_on_train_mlp
## use the same transformation (using training mean and std) on the test set
#x_test_scaled_mlp = (x_test - mean_on_train_mlp)/std_on_train_mlp

#regression_mlp = MLPRegressor(hidden_layer_sizes=[1000,1000],random_state = 16)
#regression_mlp = MLPRegressor(hidden_layer_sizes=[1,1],alpha = 1e-5,random_state = 16)
#########################
#regression_mlp = MLPRegressor(random_state = 16)
#regression_mlp = MLPRegressor(hidden_layer_sizes=[500,500], activation = 'logistic', random_state = 67)


#regression_knn = regression_knn.fit(x_train, y_train)
#regression_lr = regression_lr.fit(x_train, y_train)
#regression_rr = regression_rr.fit(x_train, y_train)
regression_rt = regression_rt.fit(x_train, y_train)
#regression_rf = regression_rf.fit(x_train, y_train)
##regression_svr = regression_svr.fit(x_train_scaled_svr, y_train)
#regression_svr = regression_svr.fit(x_train, y_train)
#regression_mlp = regression_mlp.fit(x_train, y_train)
#regression_mlp = regression_mlp.fit(x_train_scaled_mlp, y_train)


#y_pred_knn = regression_knn.predict(x_test)
#y_pred_lr = regression_lr.predict(x_test)
#y_pred_rr = regression_rr.predict(x_test)
y_pred_rt = regression_rt.predict(x_test)
#y_pred_rf = regression_rf.predict(x_test)
##y_pred_svr = regression_svr.predict(x_test_scaled_svr)
##y_pred_mlp = regression_mlp.predict(x_test)
#y_pred_mlp = regression_mlp.predict(x_test_scaled_mlp)

#mse_knn = mean_squared_error(y_pred_knn,y_test)
#mse_lr = mean_squared_error(y_pred_lr,y_test)
#mse_rr = mean_squared_error(y_pred_rr,y_test)
mse_rt = mean_squared_error(y_pred_rt,y_test)
#mse_rf = mean_squared_error(y_pred_rf,y_test)
#mse_svr = mean_squared_error(y_pred_svr,y_test)
#mse_mlp = mean_squared_error(y_pred_mlp,y_test)

#print("mse for k-nearest neighbor {}".format(mse_knn))
#print("mse for linear regression {}".format(mse_lr))
#print("mse for ridge regression {}".format(mse_rr))
print("mse for regression tree {}".format(mse_rt))
#print("mse for random forest {}".format(mse_rf))
#print("mse for support vector machine {}".format(mse_svr))
#print("mse for multilayer perceptron {}".format(mse_mlp))
#
#p_knn = plt.scatter(y_test,y_pred_knn)
#p_lr = plt.scatter(y_test,y_pred_lr)
#p_rr = plt.scatter(y_test,y_pred_rr)
p_rt = plt.scatter(y_test,y_pred_rt)
#p_rf = plt.scatter(y_test,y_pred_rf)
#p_svr = plt.scatter(y_test,y_pred_svr)
#p_mlp = plt.scatter(y_test,y_pred_mlp)

#plt.savefig('p_knn.png')
#plt.savefig('p_lr.png')
#plt.savefig('p_rr.png')
plt.savefig('p_rt.png')
#plt.savefig('p_rf.png')
#plt.savefig('p_svr.png')
#plt.savefig('p_mlp.png')