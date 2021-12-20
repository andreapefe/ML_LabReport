# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:18:03 2021

@author: andre
LAB2 - Exercie 1 - partie 2
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime #measure training time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

#Spliting dataset
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size = 0.20)

print("*********5000 Samples***************")
for i in (10, 50, 300):
    print("Model with Max iterations = ", i)
    #Artificial Neural Network classifier
    mlp= MLPClassifier(hidden_layer_sizes=(50), activation='logistic', 
                      solver='adam', #solver
                      alpha=0.01,# L2 penalty (regularization term) parameter
                      batch_size='auto', #Size of minibatches for stochastic optimizers.
                      max_iter=i, #Maximum number of iterations
                      tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change
                      verbose=False, # To Print progress messages during learning step
                      warm_start=False, #When set to True, reuse the solution of the previous call to fit as
                      early_stopping=True, #Whether to use early stopping to terminate training when
                      validation_fraction=0.1, #The proportion of training data to set aside as validation se
                      n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.)
    #had to change tolerance for convergence of the algorithm. 
    
    #training
    start_time = datetime.now()
    mlp.fit(x_train, y_train)
    end_time = datetime.now()
    
    #training time
    print("Training Time: ", (end_time-start_time))
    
    #Score calculation
    score_train = mlp.score(x_train, y_train)
    score_test = mlp.score(x_test, y_test)
    print("Score for training: {0:.3f}".format(score_train))
    print("Score for test: {0:.3f}".format(score_test))
    
    #Prediction
    y_predict = mlp.predict(x_test)
    y_predict_train = mlp.predict(x_train)
    
    #precision score
    precision = precision_score(y_test, y_predict, average='micro')
    print("Precision score: {0:.3f}".format(precision))
    
    #accuracy score
    accuracy = accuracy_score(y_test,y_predict)
    print("Accuracy score: {0:.3f}\n".format(accuracy))
    
print("\n\n")
print("************20000**********")   
sample = np.random.randint(70000, size=20000)
data = mnist.data[sample]
target = mnist.target[sample]
for i in (10, 50, 300):
    print("Model with Max iterations = ", i)
    #Artificial Neural Network classifier
    mlp= MLPClassifier(hidden_layer_sizes=(50), activation='logistic', 
                      solver='adam', #solver
                      alpha=0.01,# L2 penalty (regularization term) parameter
                      batch_size='auto', #Size of minibatches for stochastic optimizers.
                      max_iter=i, #Maximum number of iterations
                      tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change
                      verbose=False, # To Print progress messages during learning step
                      warm_start=False, #When set to True, reuse the solution of the previous call to fit as
                      early_stopping=True, #Whether to use early stopping to terminate training when
                      validation_fraction=0.1, #The proportion of training data to set aside as validation se
                      n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.)
    #had to change tolerance for convergence of the algorithm. 
    
    #training
    start_time = datetime.now()
    mlp.fit(x_train, y_train)
    end_time = datetime.now()
    
    #training time
    print("Training Time: ", (end_time-start_time))
    
    #Score calculation
    score_train = mlp.score(x_train, y_train)
    score_test = mlp.score(x_test, y_test)
    print("Score for training: {0:.3f}".format(score_train))
    print("Score for test: {0:.3f}".format(score_test))
    
    #Prediction
    y_predict = mlp.predict(x_test)
    y_predict_train = mlp.predict(x_train)
    
    #precision score
    precision = precision_score(y_test, y_predict, average='micro')
    print("Precision score: {0:.3f}".format(precision))
    
    #accuracy score
    accuracy = accuracy_score(y_test,y_predict)
    print("Accuracy score: {0:.3f}\n".format(accuracy))