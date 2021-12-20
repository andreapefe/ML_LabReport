# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:13:08 2021

@author: Andrea Pérez
TP1  Machine Learning - Exercice 2
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime #measure training time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

#Importing the dataset
dataset = np.genfromtxt("yacht_hydrodynamics.data", delimiter='')

#Dividing the input and outputs
X = dataset[:, :-1]
y = dataset[:, -1]

#dividing between training and test data
x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=0, test_size = 0.20)

#Scaling the datasets
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

neurons = ((5,5), (10,10), (20,20))
training_score = np.zeros(len(neurons))
test_score = np.zeros(len(neurons))
for i in range(0,len(neurons)):
    mlp= MLPRegressor(hidden_layer_sizes=(neurons[i]), activation='logistic',
                      solver='adam', #solver
                      alpha=0.01,# L2 penalty (regularization term) parameter
                      batch_size='auto', #Size of minibatches for stochastic optimizers.
                      learning_rate='adaptive', #Learning rate schedule for weight updates
                      learning_rate_init=0.01, #The initial learning rate used
                      max_iter=1000, #Maximum number of iterations
                      tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change
                      verbose=False, # To Print progress messages during learning step
                      warm_start=False, #When set to True, reuse the solution of the previous call to fit as
                      early_stopping=True, #Whether to use early stopping to terminate training when
                      validation_fraction=0.1, #The proportion of training data to set aside as validation se
                      n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.))
    mlp.fit(x_train, y_train);
    training_score[i] = mlp.score(x_train, y_train);
    test_score[i] = mlp.score(x_test, y_test);
    y_predict = mlp.predict(x_test)
    plt.scatter(y_predict, y_test, label="Double {0:.0f} neurons".format(neurons[i][1]))

idealx = [0, 50]
idealy = [0, 50]
plt.title("Predicted values vs target values for logistic function")
plt.plot(idealx, idealy, color='black', linestyle='dashed')
plt.legend()
plt.show()
plt.close()

training_score_relu = np.zeros(len(neurons))
test_score_relu = np.zeros(len(neurons))
for i in range(0,len(neurons)):
    mlp= MLPRegressor(hidden_layer_sizes=(neurons[i]), activation='relu',
                      solver='adam', #solver
                      alpha=0.01,# L2 penalty (regularization term) parameter
                      batch_size='auto', #Size of minibatches for stochastic optimizers.
                      learning_rate='adaptive', #Learning rate schedule for weight updates
                      learning_rate_init=0.01, #The initial learning rate used
                      max_iter=1000, #Maximum number of iterations
                      tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change
                      verbose=False, # To Print progress messages during learning step
                      warm_start=False, #When set to True, reuse the solution of the previous call to fit as
                      early_stopping=True, #Whether to use early stopping to terminate training when
                      validation_fraction=0.1, #The proportion of training data to set aside as validation se
                      n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.))
    mlp.fit(x_train, y_train);
    training_score_relu[i] = mlp.score(x_train, y_train);
    test_score_relu[i] = mlp.score(x_test, y_test);
    
    
plt.plot(training_score, label="Training score")
plt.plot(test_score, label="Test score")
positions = (0, 1, 2)
plt.title("Predicted values vs target values for relu function")
labels = ("(5,5)", "(10,10)", "(20,20)")
plt.xticks(positions, labels)
plt.plot(training_score_relu, label="Training score relu")
plt.plot(test_score_relu, label="Test score relu")
plt.legend()
plt.title("Training and test score vs number of hidden layers")
plt.show()
plt.close()


mlp= MLPRegressor(hidden_layer_sizes=(10,10,10), activation='relu', 
                  solver='adam', #solver
                  alpha=0.01,# L2 penalty (regularization term) parameter
                  batch_size='auto', #Size of minibatches for stochastic optimizers.
                  learning_rate='adaptive', #Learning rate schedule for weight updates
                  learning_rate_init=0.01, #The initial learning rate used
                  max_iter=1000, #Maximum number of iterations
                  tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change
                  verbose=False, # To Print progress messages during learning step
                  warm_start=False, #When set to True, reuse the solution of the previous call to fit as
                  early_stopping=True, #Whether to use early stopping to terminate training when
                  validation_fraction=0.1, #The proportion of training data to set aside as validation se
                  n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.)
#had to change tolerance for convergence of the algorithm. 
start_time = datetime.now()
mlp.fit(x_train, y_train)
end_time = datetime.now()

#training score
score_train = mlp.score(x_train, y_train)
print("Training score : {0:.3f}".format(score_train))
#test score
score_test = mlp.score(x_test, y_test)
print("Test score : ", score_test)
#training time
print("Training Time : ", (end_time-start_time))

#predicted vs target values
y_predict = mlp.predict(x_test)
y_predict_train = mlp.predict(x_train)

plt.plot(y_predict ,'bx', label="predict")
plt.plot(y_test, 'ro', label="target")
plt.title("Predicted and target values for test data")
plt.legend()
plt.show()
plt.close()
plt.plot(y_predict_train ,'bx', label="predict")
plt.plot(y_train, 'ro', label="target")
plt.title("Predicted and target values for training data")
plt.legend()
plt.show()
plt.close()

