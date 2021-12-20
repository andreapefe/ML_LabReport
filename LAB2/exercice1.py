# -*- coding: utf-8 -*-
"""
Andrea Pérez

LAB2 - Exercie 1 - partie 1
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

'''
Questions de preparation
#print(mnist) # toutes les infos sur le dataset
print (mnist.data) #array de data
print (mnist.target) #array des target values
len(mnist.data) #longueur 
help(len)    #Aide fonction
print (mnist.data.shape) # print taille de la dataset sans target (X)
print (mnist.target.shape)  # # print taille de la dataset pour target (Y)
mnist.data[0]
mnist.data[0][1]
mnist.data[:,1]
mnist.data[:100]

#Reconstruction de l'image
images = mnist.data.reshape((-1, 28, 28)) # à partir des 784 pixel données 
#                                           on reconstruit un tableau
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")#à partir du tableau -> image
plt.show()
'''

#Spliting dataset
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size = 0.20)

#Artificial Neural Network regressor
mlp= MLPClassifier(hidden_layer_sizes=(50), activation='logistic', 
                  solver='adam', #solver
                  alpha=0.01,# L2 penalty (regularization term) parameter
                  batch_size='auto', #Size of minibatches for stochastic optimizers.
                  max_iter=1000, #Maximum number of iterations
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
images = x_test.reshape((-1, 28, 28)) # à partir des 784 pixel données 
#                                           on reconstruit un tableau
plt.imshow(images[4],cmap=plt.cm.gray_r,interpolation="nearest")#à partir du tableau -> image
plt.show()
print("Class of image 4: ", y_test[4])
print("Predicted class of image 4: ", y_predict[4])

#precision score
precision_score(y_test, y_predict, average='micro')

#accuracy score
accuracy_score(y_test,y_predict)

#confusion matrix
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot()
