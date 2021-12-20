# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:50:28 2021

@author: andrea
LAB2 - Exercice 2 
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime #measure training time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# fetch dataset from openml (might take some time)
mnist = fetch_openml('mnist_784', as_frame=False)

sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

#trainig and test set
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size = 0.20)

#%%
neurons =[]
time = []
score_train = np.zeros(5)
score_test = np.zeros(5)
accuracy = np.zeros(5)
precision = np.zeros(5)

for i in range(0,5):
    random = np.random.randint(50,100)
    neurons = list(neurons)
    neurons.append(random)
    neurons = tuple(neurons)
    
    print("Model with {0} neurons ".format(neurons))
    #Artificial Neural Network classifier
    mlp= MLPClassifier(hidden_layer_sizes=(neurons), activation='logistic', 
                      solver='adam', #solver
                      alpha=0.01,# L2 penalty (regularization term) parameter
                      batch_size='auto', #Size of minibatches for stochastic optimizers.
                      max_iter=300, #Maximum number of iterations
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
    time.append((end_time-start_time).total_seconds())
    
    #Score calculation
    score_train[i] = mlp.score(x_train, y_train)
    score_test[i] = mlp.score(x_test, y_test)
    #print("Score for training: {0:.3f}".format(score_train))
    #print("Score for test: {0:.3f}".format(score_test))
    
    #Prediction
    y_predict = mlp.predict(x_test)
    y_predict_train = mlp.predict(x_train)
    
    #precision score
    precision[i] = precision_score(y_test, y_predict, average='micro')
    #print("Precision score: {0:.3f}".format(precision))
    
    #accuracy score
    accuracy[i] = accuracy_score(y_test,y_predict)
    #print("Accuracy score: {0:.3f}\n".format(accuracy))
    
    cm = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
    disp.plot()
    plt.title("Hidden layers {0} ".format(i))
    plt.show()
    plt.close()

#%%
n = np.arange(1,6)
plt.plot(n,score_train, label="Train")
plt.plot(n, score_test, label="Test")
plt.title("Training and test score, precision and accuracy number of hiden layers")
plt.plot(n, accuracy,'o', label="Accuracy")
plt.plot(n, precision,'x', label="Precision")
plt.legend()
plt.show()
plt.close()

plt.plot(n,time)
plt.ylabel("Temps(s)")
plt.xlabel("NUmber of hidden layers")
plt.title("Time vs number of hidden layers")
plt.show()
plt.close()